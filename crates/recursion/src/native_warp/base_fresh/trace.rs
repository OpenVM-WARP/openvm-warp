use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    hasher::MerkleHasher,
    warp_accum::{
        BinaryMerkleMultiproofRecord, MerkleCompressionRecord, MerkleNodeOrigin,
        StackedRsBatchOpeningVerification, StackedRsFreshCommitment,
    },
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    generate_native_direct_fresh_commitment_trace, generate_native_direct_fresh_projection_trace,
    generate_native_direct_fresh_root_trace, NativeDirectFreshCommitmentTraceInput,
    NativeDirectFreshProjectionCols, NativeDirectFreshProjectionRecord,
    NativeDirectFreshRootTraceInput,
};
use crate::native_warp::{
    generate_native_leaf_hash_trace, generate_native_merkle_leaf_adapter_trace,
    generate_native_merkle_multiproof_trace, NativeLeafHashInput, NativeLeafHashTrace,
    NativeMerkleCompressionCols, NativeMerkleLeafAdapterInput,
};

/// Complete history witness for one direct-original-root VACC batch.
///
/// Each application root remains the authenticated oracle. The projection
/// trace combines only the verifier-selected cells, so this path introduces
/// neither a scalar Merkle root nor a second RS encoding.
pub struct NativeDirectFreshOpeningTraces {
    pub commitment: RowMajorMatrix<F>,
    pub roots: RowMajorMatrix<F>,
    /// Source-major projection rows split only at source boundaries. Each
    /// matrix is paired with one `NativeDirectFreshProjectionAir` whose first
    /// source is fixed in the verifier key.
    pub projections: Vec<RowMajorMatrix<F>>,
    pub leaf_hash: NativeLeafHashTrace,
    pub merkle: RowMajorMatrix<F>,
    pub leaf_adapter: RowMajorMatrix<F>,
    pub compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

#[allow(clippy::too_many_arguments)]
pub fn generate_native_direct_fresh_opening_traces<H>(
    hasher: &H,
    commitments: &[StackedRsFreshCommitment<EF, Digest>],
    openings: &[StackedRsBatchOpeningVerification<F, EF, Digest>],
    commitment_tidxs: &[usize],
    proof_idx: usize,
    max_fresh: usize,
    max_roots: usize,
    first_tree_id: u32,
    root_tree_stride: u32,
    projection_sources_per_shard: usize,
    max_projection_height: usize,
) -> Option<NativeDirectFreshOpeningTraces>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    let first_commitment = commitments.first()?;
    let first_opening = openings.first()?;
    let shift_count = first_opening.flat_indices.len();
    if commitments.len() > max_fresh
        || commitments.len() != openings.len()
        || commitments.len() != commitment_tidxs.len()
        || shift_count == 0
        || max_roots == 0
        || projection_sources_per_shard == 0
        || max_projection_height == 0
        || !max_projection_height.is_power_of_two()
        || root_tree_stride as usize != 1usize.checked_add(shift_count)?
    {
        return None;
    }
    let tree_source_stride = max_roots.checked_mul(root_tree_stride as usize)?;
    let mut commitment_inputs = Vec::with_capacity(commitments.len());
    let mut root_inputs = Vec::with_capacity(commitments.len());
    let mut projection_records = Vec::new();
    let mut leaf_storage = Vec::<(u32, u32, Vec<F>, Vec<u32>)>::new();
    let mut inner_records = Vec::<(u32, BinaryMerkleMultiproofRecord<Digest>)>::new();
    let mut adapters = Vec::new();

    for (source, ((commitment, opening), &commitment_tidx)) in commitments
        .iter()
        .zip(openings)
        .zip(commitment_tidxs)
        .enumerate()
    {
        let rows_per_query = commitment.rows_per_query;
        let codeword_len = 1usize.checked_shl(commitment.log_codeword_len as u32)?;
        let query_stride = codeword_len.checked_div(rows_per_query)?;
        let source_tree_id =
            (first_tree_id as usize).checked_add(source.checked_mul(tree_source_stride)?)?;
        if !opening.recorded
            || commitment.roots.is_empty()
            || commitment.roots.len() > max_roots
            || commitment.roots.len() != commitment.widths.len()
            || commitment.widths.contains(&0)
            || commitment.log_message_len != first_commitment.log_message_len
            || commitment.log_codeword_len != first_commitment.log_codeword_len
            || rows_per_query != first_commitment.rows_per_query
            || rows_per_query == 0
            || !rows_per_query.is_power_of_two()
            || query_stride == 0
            || !query_stride.is_power_of_two()
            || opening.theta != commitment.theta
            || opening.flat_indices != first_opening.flat_indices
            || opening.values.len() != shift_count
            || opening.roots.len() != commitment.roots.len()
        {
            return None;
        }
        commitment_inputs.push(NativeDirectFreshCommitmentTraceInput {
            source_active: true,
            proof_idx,
            commitment_tidx,
            root_count: commitment.roots.len(),
            l_skip: commitment.l_skip,
            native_log_message_len: commitment.native_log_message_len,
            log_message_len: commitment.log_message_len,
            log_codeword_len: commitment.log_codeword_len,
            rows_per_query,
            theta: commitment.theta,
            first_tree_id: source_tree_id,
        });
        root_inputs.push(NativeDirectFreshRootTraceInput {
            source_active: true,
            root_count: commitment.roots.len(),
            proof_idx,
            commitment_tidx,
            first_tree_id: source_tree_id,
            theta: commitment.theta,
            roots: commitment.roots.clone(),
            widths: commitment.widths.clone(),
        });

        for (root_ordinal, ((&root, &width), root_opening)) in commitment
            .roots
            .iter()
            .zip(&commitment.widths)
            .zip(&opening.roots)
            .enumerate()
        {
            let outer_tree_id =
                source_tree_id.checked_add(root_ordinal.checked_mul(root_tree_stride as usize)?)?;
            if root_opening.root != root
                || root_opening.width as usize != width
                || root_opening.opened_rows.len() != shift_count
                || root_opening.query_indices.len() != shift_count
                || root_opening.query_digests.len() != shift_count
                || root_opening.multiproof.expected_root != root
                || root_opening.multiproof.depth as usize != query_stride.ilog2() as usize
                || root_opening.multiproof.leaf_indices != root_opening.query_indices
                || root_opening.multiproof.leaf_digests != root_opening.query_digests
            {
                return None;
            }
            for shift in 0..shift_count {
                let flat_index = opening.flat_indices[shift] as usize;
                if flat_index >= codeword_len {
                    return None;
                }
                let query_index = flat_index % query_stride;
                let row_offset = flat_index / query_stride;
                let rows = &root_opening.opened_rows[shift];
                if root_opening.query_indices[shift] as usize != query_index
                    || rows.len() != rows_per_query
                    || rows.iter().any(|row| row.len() != width)
                    || row_offset >= rows_per_query
                {
                    return None;
                }
                let inner_tree_id = outer_tree_id.checked_add(1 + shift)?;
                let leaf_digests = rows
                    .iter()
                    .map(|row| hasher.hash_slice(row))
                    .collect::<Vec<_>>();
                let inner = complete_binary_tree_record(hasher, leaf_digests)?;
                if inner.expected_root != root_opening.query_digests[shift] {
                    return None;
                }
                for (leaf_index, row) in rows.iter().enumerate() {
                    leaf_storage.push((
                        u32::try_from(inner_tree_id).ok()?,
                        u32::try_from(leaf_index).ok()?,
                        row.clone(),
                        vec![u32::from(leaf_index == row_offset); width],
                    ));
                }
                adapters.push(NativeMerkleLeafAdapterInput {
                    proof_idx: proof_idx.try_into().ok()?,
                    bypass: rows_per_query == 1,
                    outer_multiplicity: 1,
                    inner_tree_id: u32::try_from(inner_tree_id).ok()?,
                    outer_tree_id: u32::try_from(outer_tree_id).ok()?,
                    query_index: root_opening.query_indices[shift],
                    inner_depth: rows_per_query.ilog2(),
                    digest: inner.expected_root,
                });
                inner_records.push((u32::try_from(inner_tree_id).ok()?, inner));
            }
        }

        for shift in 0..shift_count {
            let flat_index = opening.flat_indices[shift] as usize;
            let query_index = flat_index % query_stride;
            let row_offset = flat_index / query_stride;
            let mut theta_power = EF::ONE;
            let mut accumulated = EF::ZERO;
            for (root_ordinal, root_opening) in opening.roots.iter().enumerate() {
                let width = commitment.widths[root_ordinal];
                let tree_id = source_tree_id
                    .checked_add(root_ordinal.checked_mul(root_tree_stride as usize)?)?;
                let row = root_opening.opened_rows.get(shift)?.get(row_offset)?;
                for (column, &base_value) in row.iter().enumerate() {
                    let accumulated_before = accumulated;
                    accumulated += theta_power * EF::from(base_value);
                    projection_records.push(NativeDirectFreshProjectionRecord {
                        source,
                        shift,
                        root_ordinal,
                        root_count: commitment.roots.len(),
                        column,
                        width,
                        first_tree_id: source_tree_id,
                        tree_id,
                        root: commitment.roots[root_ordinal],
                        commitment_tidx,
                        flat_index,
                        query_index,
                        row_offset,
                        theta: commitment.theta,
                        theta_power,
                        accumulated_before,
                        base_value,
                        accumulated_after: accumulated,
                    });
                    theta_power *= commitment.theta;
                }
            }
            if accumulated != opening.values[shift] {
                return None;
            }
        }
    }

    let proof_idx_u32 = u32::try_from(proof_idx).ok()?;
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
    let mut merkle_refs = inner_records
        .iter()
        .filter(|(_, record)| !record.compressions.is_empty())
        .map(|(tree_id, record)| (*tree_id, record))
        .collect::<Vec<_>>();
    for (source, opening) in openings.iter().enumerate() {
        let source_tree_id =
            (first_tree_id as usize).checked_add(source.checked_mul(tree_source_stride)?)?;
        for (root_ordinal, root) in opening.roots.iter().enumerate() {
            merkle_refs.push((
                u32::try_from(
                    source_tree_id
                        .checked_add(root_ordinal.checked_mul(root_tree_stride as usize)?)?,
                )
                .ok()?,
                &root.multiproof,
            ));
        }
    }
    let compression_inputs = merkle_refs
        .iter()
        .flat_map(|(_, record)| {
            record.compressions.iter().map(|compression| {
                core::array::from_fn(|index| {
                    if index < compression.left.len() {
                        compression.left[index]
                    } else {
                        compression.right[index - compression.left.len()]
                    }
                })
            })
        })
        .collect();
    let merkle = generate_native_merkle_multiproof_trace(proof_idx, &merkle_refs, None)
        .unwrap_or_else(zero_merkle_trace);
    let leaf_adapter = generate_native_merkle_leaf_adapter_trace(
        &adapters,
        first_commitment.rows_per_query.ilog2() as usize,
        None,
    )?;
    let projection_shard_count = max_fresh.div_ceil(projection_sources_per_shard);
    let projection_width = NativeDirectFreshProjectionCols::<F>::width();
    let mut projections = Vec::with_capacity(projection_shard_count);
    for shard in 0..projection_shard_count {
        let first_source = shard.checked_mul(projection_sources_per_shard)?;
        let source_end = first_source
            .checked_add(projection_sources_per_shard)?
            .min(max_fresh);
        let first_record =
            projection_records.partition_point(|record| record.source < first_source);
        let last_record = projection_records.partition_point(|record| record.source < source_end);
        let records = &projection_records[first_record..last_record];
        let projection = if records.is_empty() {
            // The AIR explicitly permits an empty suffix shard. A one-row zero
            // trace has no lookup multiplicity and cannot hide an active source,
            // because the commitment/source bus fixes the complete multiset.
            RowMajorMatrix::new(F::zero_vec(projection_width), projection_width)
        } else {
            if records[0].source != first_source {
                return None;
            }
            generate_native_direct_fresh_projection_trace(proof_idx, records, shift_count, None)?
        };
        if projection.height() > max_projection_height {
            return None;
        }
        projections.push(projection);
    }
    if projection_records
        .last()
        .is_some_and(|record| record.source >= max_fresh)
    {
        return None;
    }

    Some(NativeDirectFreshOpeningTraces {
        commitment: generate_native_direct_fresh_commitment_trace(
            &commitment_inputs,
            max_fresh,
            max_roots,
            tree_source_stride,
            None,
        )?,
        roots: generate_native_direct_fresh_root_trace(
            &root_inputs,
            max_fresh,
            max_roots,
            root_tree_stride as usize,
            tree_source_stride,
            None,
        )?,
        projections,
        leaf_hash,
        merkle,
        leaf_adapter,
        compression_inputs,
    })
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
        vec![F::ZERO; NativeMerkleCompressionCols::<F>::width()],
        NativeMerkleCompressionCols::<F>::width(),
    )
}

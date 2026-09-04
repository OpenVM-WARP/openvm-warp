use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    hasher::MerkleHasher,
    interaction::InteractionBuilder,
    transcript::TranscriptLog,
    warp_accum::{
        terminal_whir::{terminal_whir_query_root, TerminalWhirLayout},
        verify_binary_merkle_multiproof_recorded, BinaryMerkleMultiProof,
        BinaryMerkleMultiproofRecord, TerminalWhirVerification,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    native_warp::{
        bus::{
            NativeLeafValueBus, NativeLeafValueMessage, NativeOpeningLeafBus,
            NativeOpeningLeafMessage,
        },
        terminal::{
            NativeTerminalWhirFoldingBus, NativeTerminalWhirFoldingMessage,
            NativeTerminalWhirQueryBus, NativeTerminalWhirQueryMessage,
        },
        NativeMerkleLeafAdapterInput,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTerminalWhirOpenedCols<T> {
    pub active: T,
    pub round: T,
    pub query: T,
    pub global_query: T,
    pub coset: T,
    pub is_first: T,
    pub is_first_in_round: T,
    pub is_first_in_query: T,
    pub inner_tree_id: T,
    pub outer_tree_id: T,
    pub merkle_index_sample: T,
    pub merkle_index: T,
    pub zi_root: T,
    pub zi: T,
    pub yi: [T; D_EF],
    pub twiddle: T,
    pub value: [T; D_EF],
    pub query_digest: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(NativeTerminalWhirOpenedCols<u8>)]
pub struct NativeTerminalWhirOpenedAir {
    pub query_bus: NativeTerminalWhirQueryBus,
    pub folding_bus: NativeTerminalWhirFoldingBus,
    pub leaf_value_bus: NativeLeafValueBus,
    pub opening_leaf_bus: NativeOpeningLeafBus,
    pub k: usize,
    /// Zero for a vector-alphabet leaf. `k` for the coefficient-subgroup
    /// layout, whose one outer leaf is an inner tree of `2^k` scalar rows.
    pub inner_depth: usize,
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirOpenedAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirOpenedAir {}
impl BaseAir<F> for NativeTerminalWhirOpenedAir {
    fn width(&self) -> usize {
        NativeTerminalWhirOpenedCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirOpenedAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        debug_assert!(self.k > 0);
        let main = builder.main();
        let local = main.row_slice(0).expect("native terminal WHIR opened row");
        let next = main
            .row_slice(1)
            .expect("native terminal WHIR next opened row");
        let local: &NativeTerminalWhirOpenedCols<AB::Var> = (*local).borrow();
        let next: &NativeTerminalWhirOpenedCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<3>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.active.into(),
                    counter: [local.round.into(), local.query.into(), local.coset.into()],
                    is_first: [
                        local.is_first_in_round.into(),
                        local.is_first_in_query.into(),
                        local.active.into(),
                    ],
                },
                NestedForLoopIoCols {
                    is_enabled: next.active.into(),
                    counter: [next.round.into(), next.query.into(), next.coset.into()],
                    is_first: [
                        next.is_first_in_round.into(),
                        next.is_first_in_query.into(),
                        next.active.into(),
                    ],
                },
            ),
        );
        builder.assert_bool(local.is_first);
        builder
            .when_first_row()
            .assert_eq(local.is_first, local.active);
        let same_query = next.active - next.is_first_in_query;
        builder
            .when(local.active - same_query.clone())
            .assert_eq(local.coset, AB::Expr::from_usize((1usize << self.k) - 1));
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_query.clone());
        same.assert_eq(next.global_query, local.global_query);
        same.assert_eq(next.inner_tree_id, local.inner_tree_id);
        same.assert_eq(next.outer_tree_id, local.outer_tree_id);
        same.assert_eq(next.merkle_index_sample, local.merkle_index_sample);
        same.assert_eq(next.merkle_index, local.merkle_index);
        same.assert_eq(next.zi_root, local.zi_root);
        same.assert_eq(next.zi, local.zi);
        if self.inner_depth == 0 {
            same.assert_eq(next.twiddle, local.twiddle);
        } else {
            let omega = AB::F::two_adic_generator(self.k);
            same.assert_eq(next.twiddle, AB::Expr::from(local.twiddle) * omega);
        }
        assert_array_eq(&mut same, next.yi, local.yi);
        assert_array_eq(&mut same, next.query_digest, local.query_digest);
        if self.inner_depth == 0 {
            builder
                .when(local.is_first_in_query)
                .assert_one(local.twiddle);
        } else {
            // Scalar coefficient-subgroup rows are evaluations at
            // `zi_root * omega^coset`. In particular the second half carries
            // the negated twiddles needed by binary WHIR folding.
            builder
                .when(local.is_first_in_query)
                .assert_eq(local.twiddle, local.zi_root);
        }

        self.query_bus.receive(
            builder,
            NativeTerminalWhirQueryMessage {
                round: local.round.into(),
                query: local.query.into(),
                global_query: local.global_query.into(),
                inner_tree_id: local.inner_tree_id.into(),
                outer_tree_id: local.outer_tree_id.into(),
                merkle_index_sample: local.merkle_index_sample.into(),
                merkle_index: local.merkle_index.into(),
                zi_root: local.zi_root.into(),
                zi: local.zi.into(),
                yi: local.yi.map(Into::into),
            },
            local.is_first_in_query,
        );
        for (position, value) in local.value.into_iter().enumerate() {
            self.leaf_value_bus.lookup_key(
                builder,
                NativeLeafValueMessage {
                    proof_idx: AB::Expr::ZERO,
                    tree_id: local.inner_tree_id.into(),
                    index: if self.inner_depth == 0 {
                        AB::Expr::ZERO
                    } else {
                        local.coset.into()
                    },
                    position: if self.inner_depth == 0 {
                        local.coset * AB::Expr::from_usize(D_EF) + AB::Expr::from_usize(position)
                    } else {
                        AB::Expr::from_usize(position)
                    },
                    value: value.into(),
                },
                local.active,
            );
        }
        if self.inner_depth == 0 {
            // The vector alphabet is hashed as one wide leaf.
            self.opening_leaf_bus.receive(
                builder,
                NativeOpeningLeafMessage {
                    proof_idx: AB::Expr::ZERO,
                    tree_id: local.inner_tree_id.into(),
                    index: AB::Expr::ZERO,
                    digest: local.query_digest.map(Into::into),
                },
                local.is_first_in_query,
            );
            self.opening_leaf_bus.send(
                builder,
                NativeOpeningLeafMessage {
                    proof_idx: AB::Expr::ZERO,
                    tree_id: local.outer_tree_id.into(),
                    index: local.merkle_index.into(),
                    digest: local.query_digest.map(Into::into),
                },
                local.is_first_in_query,
            );
        } else {
            // The leaf adapter sends two authenticated copies of the inner
            // subtree root: one to the outer Merkle proof and one here, tying
            // the folding/query record to that exact root.
            self.opening_leaf_bus.receive(
                builder,
                NativeOpeningLeafMessage {
                    proof_idx: AB::Expr::ZERO,
                    tree_id: local.outer_tree_id.into(),
                    index: local.merkle_index.into(),
                    digest: local.query_digest.map(Into::into),
                },
                local.is_first_in_query,
            );
        }
        self.folding_bus.send(
            builder,
            NativeTerminalWhirFoldingMessage {
                round: local.round.into(),
                query: local.query.into(),
                height: AB::Expr::ZERO,
                coset_shift: local.zi_root.into(),
                coset_size: AB::Expr::from_usize(1usize << self.k),
                coset_index: local.coset.into(),
                twiddle: local.twiddle.into(),
                value: local.value.map(Into::into),
                z_final: local.zi.into(),
                y_final: local.yi.map(Into::into),
            },
            local.active,
        );
    }
}

#[derive(Clone, Debug)]
pub struct NativeTerminalOwnedLeafHashInput {
    pub tree_id: u32,
    pub leaf_index: u32,
    pub values: Vec<F>,
    pub lookup_counts: Vec<u32>,
}

pub struct NativeTerminalWhirOpenedTrace {
    pub matrix: RowMajorMatrix<F>,
    pub leaves: Vec<NativeTerminalOwnedLeafHashInput>,
    pub inner_merkle: Vec<(u32, BinaryMerkleMultiproofRecord<Digest>)>,
    pub leaf_adapters: Vec<NativeMerkleLeafAdapterInput>,
}

#[allow(clippy::too_many_arguments)]
pub fn generate_native_terminal_whir_opened_trace<H>(
    hasher: &H,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    k: usize,
    initial_log_domain_size: usize,
    inner_tree_id_offset: usize,
    outer_tree_id_offset: usize,
    required_height: Option<usize>,
    layout: TerminalWhirLayout,
) -> Option<NativeTerminalWhirOpenedTrace>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    if k == 0 || initial_log_domain_size <= k {
        return None;
    }
    let coset_size = 1usize.checked_shl(k as u32)?;
    let query_count = verification
        .rounds
        .iter()
        .map(|round| round.query_indices.len())
        .sum::<usize>();
    let valid_rows = query_count.checked_mul(coset_size)?;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTerminalWhirOpenedCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut leaves = Vec::with_capacity(query_count);
    let mut inner_merkle = Vec::new();
    let mut leaf_adapters = Vec::new();
    let mut row_index = 0usize;
    let mut global_query = 0usize;
    for (round_index, round) in verification.rounds.iter().enumerate() {
        let query_tidx = round
            .transcript_span
            .end
            .operations
            .checked_sub(D_EF + round.query_indices.len())?;
        if round.query_indices.len() != round.opened_rows.len()
            || round.query_indices.len() != round.query_digests.len()
            || round.query_indices.len() != round.query_roots.len()
            || round.query_indices.len() != round.folded_values.len()
        {
            return None;
        }
        for query_index in 0..round.query_indices.len() {
            // Vector-alphabet WHIR stores one row of width `2^k`; the
            // coefficient-subgroup terminal stores `2^k` scalar rows in one
            // Merkle leaf. Both hash and fold the same ordered extension-field
            // values, so normalize only the recorded view—never reconstruct
            // or recommit the codeword.
            let opened_rows = &round.opened_rows[query_index];
            let scalar_rows =
                opened_rows.len() == coset_size && opened_rows.iter().all(|row| row.len() == 1);
            let opened_values = if let [opened_row] = opened_rows.as_slice() {
                (opened_row.len() == coset_size).then(|| opened_row.clone())?
            } else if scalar_rows {
                opened_rows.iter().map(|row| row[0]).collect::<Vec<_>>()
            } else {
                return None;
            };
            let inner_tree_id = u32::try_from(inner_tree_id_offset + global_query).ok()?;
            let outer_tree_id = u32::try_from(outer_tree_id_offset + round_index).ok()?;
            if scalar_rows {
                let row_digests = opened_values
                    .iter()
                    .map(|value| hasher.hash_slice(value.as_basis_coefficients_slice()))
                    .collect::<Vec<_>>();
                let root = hasher.tree_compress(row_digests.clone());
                if root != round.query_digests[query_index] {
                    return None;
                }
                let indices = (0..coset_size).collect::<Vec<_>>();
                let record = verify_binary_merkle_multiproof_recorded(
                    hasher,
                    root,
                    &indices,
                    &row_digests,
                    k,
                    &BinaryMerkleMultiProof { siblings: vec![] },
                )
                .ok()?;
                inner_merkle.push((inner_tree_id, record));
                for (coset, value) in opened_values.iter().enumerate() {
                    leaves.push(NativeTerminalOwnedLeafHashInput {
                        tree_id: inner_tree_id,
                        leaf_index: u32::try_from(coset).ok()?,
                        values: value.as_basis_coefficients_slice().to_vec(),
                        lookup_counts: vec![1; D_EF],
                    });
                }
                leaf_adapters.push(NativeMerkleLeafAdapterInput {
                    proof_idx: 0,
                    outer_multiplicity: 2,
                    inner_tree_id,
                    outer_tree_id,
                    query_index: round.query_indices[query_index],
                    inner_depth: u32::try_from(k).ok()?,
                    bypass: false,
                    digest: root,
                });
            } else {
                let limbs = opened_values
                    .iter()
                    .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
                    .collect::<Vec<_>>();
                if hasher.hash_slice(&limbs) != round.query_digests[query_index] {
                    return None;
                }
                leaves.push(NativeTerminalOwnedLeafHashInput {
                    tree_id: inner_tree_id,
                    leaf_index: 0,
                    values: limbs,
                    lookup_counts: vec![1; coset_size * D_EF],
                });
            }
            // The verification record stores the post-fold point. Scalar
            // openings instead live at `x * <omega_k>`, so compile `x` from
            // the exact authenticated query index and descriptor-bound
            // layout. This parity-aware map is essential for the initial
            // coefficient two-coset oracle.
            let scalar_root = if scalar_rows {
                Some(
                    terminal_whir_query_root::<F>(
                        layout,
                        round_index == 0,
                        round.query_indices[query_index] as usize,
                        usize::try_from(round.log_rs_domain_size).ok()?,
                        k,
                    )
                    .ok()?,
                )
            } else {
                None
            };
            let scalar_omega = scalar_rows.then(|| F::two_adic_generator(k));
            for (coset, &value) in opened_values.iter().enumerate() {
                let row = &mut values[row_index * width..(row_index + 1) * width];
                let cols: &mut NativeTerminalWhirOpenedCols<F> = row.borrow_mut();
                cols.active = F::ONE;
                cols.round = F::from_usize(round_index);
                cols.query = F::from_usize(query_index);
                cols.global_query = F::from_usize(global_query);
                cols.coset = F::from_usize(coset);
                cols.is_first = F::from_bool(row_index == 0);
                cols.is_first_in_round = F::from_bool(query_index == 0 && coset == 0);
                cols.is_first_in_query = F::from_bool(coset == 0);
                cols.inner_tree_id = F::from_u32(inner_tree_id);
                cols.outer_tree_id = F::from_u32(outer_tree_id);
                cols.merkle_index_sample = *transcript
                    .values()
                    .get(query_tidx.checked_add(query_index)?)?;
                cols.merkle_index = F::from_u32(round.query_indices[query_index]);
                cols.zi_root = scalar_root.unwrap_or(round.query_roots[query_index]);
                cols.zi = round.query_roots[query_index];
                copy_ext(&mut cols.yi, round.folded_values[query_index]);
                cols.twiddle = match (scalar_root, scalar_omega) {
                    (Some(root), Some(omega)) => root * omega.exp_u64(coset as u64),
                    _ => F::ONE,
                };
                copy_ext(&mut cols.value, value);
                cols.query_digest = round.query_digests[query_index];
                row_index += 1;
            }
            global_query += 1;
        }
    }
    Some(NativeTerminalWhirOpenedTrace {
        matrix: RowMajorMatrix::new(values, width),
        leaves,
        inner_merkle,
        leaf_adapters,
    })
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

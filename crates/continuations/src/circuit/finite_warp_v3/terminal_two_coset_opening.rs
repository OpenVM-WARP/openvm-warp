//! Scalar opening adapter for the production coefficient-two-coset terminal
//! WHIR oracle.
//!
//! The legacy terminal opening AIR interprets one authenticated leaf as a
//! width-`2^k` vector alphabet.  Production `CoefficientTwoCosetGrs` instead
//! authenticates `2^k` consecutive scalar leaves and compresses those leaves
//! into one depth-`k` subtree root before the outer Merkle proof.  This module
//! changes only that projection.  The leaf hash and Merkle compression AIRs
//! remain the recursive lane's existing, exact implementations.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_recursion_circuit::native_warp::{
    NativeLeafValueBus, NativeLeafValueMessage, NativeMerkleRootBus, NativeMerkleRootMessage,
    NativeOpeningLeafBus, NativeOpeningLeafMessage, NativeTerminalOwnedLeafHashInput,
    NativeTerminalWhirFoldingBus, NativeTerminalWhirFoldingMessage, NativeTerminalWhirOpenedCols,
    NativeTerminalWhirQueryBus, NativeTerminalWhirQueryMessage,
};
use openvm_stark_backend::{
    hasher::MerkleHasher,
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{
        extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing, PrimeField32,
        TwoAdicField,
    },
    transcript::TranscriptLog,
    warp_accum::{
        terminal_whir::{terminal_whir_query_root, TerminalWhirLayout},
        verify_binary_merkle_multiproof_recorded, BinaryMerkleMultiProof,
        BinaryMerkleMultiproofRecord, TerminalWhirVerification,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{FiniteWarpV3TwoCosetTerminalProfile, FINITE_WARP_V3_TWO_COSET_K};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3TwoCosetOpeningError {
    UnsupportedProfile,
    VerificationShape,
    Transcript,
    QueryIndex,
    QueryRoot,
    OpeningShape,
    MerkleRoot,
    TreeId,
    TraceHeight,
}

impl core::fmt::Display for FiniteWarpV3TwoCosetOpeningError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "finite WARP v3 terminal two-coset opening: {self:?}"
        )
    }
}

impl std::error::Error for FiniteWarpV3TwoCosetOpeningError {}

/// Production scalar-opening AIR.  Its columns intentionally remain the
/// terminal recursive lane's established opening columns; only the leaf and
/// folding interpretations differ.
#[derive(Clone, Debug)]
pub struct FiniteWarpV3TwoCosetTerminalWhirOpenedAir {
    pub query_bus: NativeTerminalWhirQueryBus,
    pub folding_bus: NativeTerminalWhirFoldingBus,
    pub leaf_value_bus: NativeLeafValueBus,
    pub opening_leaf_bus: NativeOpeningLeafBus,
    pub merkle_root_bus: NativeMerkleRootBus,
    pub k: usize,
}

impl FiniteWarpV3TwoCosetTerminalWhirOpenedAir {
    pub fn new(
        profile: &FiniteWarpV3TwoCosetTerminalProfile,
        query_bus: NativeTerminalWhirQueryBus,
        folding_bus: NativeTerminalWhirFoldingBus,
        leaf_value_bus: NativeLeafValueBus,
        opening_leaf_bus: NativeOpeningLeafBus,
        merkle_root_bus: NativeMerkleRootBus,
    ) -> Result<Self, FiniteWarpV3TwoCosetOpeningError> {
        if profile.whir_k() != FINITE_WARP_V3_TWO_COSET_K
            || profile.alpha_len() != profile.log_message_len() + 1
            || profile.log_message_len() > F::TWO_ADICITY
        {
            return Err(FiniteWarpV3TwoCosetOpeningError::UnsupportedProfile);
        }
        Ok(Self {
            query_bus,
            folding_bus,
            leaf_value_bus,
            opening_leaf_bus,
            merkle_root_bus,
            k: FINITE_WARP_V3_TWO_COSET_K,
        })
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TwoCosetTerminalWhirOpenedAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TwoCosetTerminalWhirOpenedAir {}
impl BaseAir<F> for FiniteWarpV3TwoCosetTerminalWhirOpenedAir {
    fn width(&self) -> usize {
        NativeTerminalWhirOpenedCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3TwoCosetTerminalWhirOpenedAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        BinomiallyExtendable<{ D_EF }> + TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("finite two-coset terminal opened row");
        let next_row = main
            .row_slice(1)
            .expect("finite two-coset terminal next opened row");
        let local: &NativeTerminalWhirOpenedCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTerminalWhirOpenedCols<AB::Var> = (*next_row).borrow();

        for flag in [
            local.active,
            local.is_first,
            local.is_first_in_round,
            local.is_first_in_query,
        ] {
            builder.assert_bool(flag);
        }
        builder
            .when_first_row()
            .assert_eq(local.is_first, local.active);
        builder
            .when_first_row()
            .when(local.active)
            .assert_zero(local.round);
        builder
            .when_first_row()
            .when(local.active)
            .assert_zero(local.query);
        builder
            .when_first_row()
            .when(local.active)
            .assert_zero(local.global_query);
        builder
            .when_first_row()
            .when(local.active)
            .assert_zero(local.coset);

        let same_query = next.active - next.is_first_in_query;
        let query_end = local.active - same_query.clone();
        builder
            .when(query_end)
            .assert_eq(local.coset, AB::Expr::from_usize((1usize << self.k) - 1));
        builder
            .when(local.is_first_in_query)
            .assert_zero(local.coset);
        builder
            .when(local.is_first_in_query)
            .assert_one(local.twiddle);

        let omega_k = AB::Expr::from_prime_subfield(
            <<AB::Expr as PrimeCharacteristicRing>::PrimeSubfield as TwoAdicField>::two_adic_generator(
                self.k,
            ),
        );
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_query);
        same.assert_eq(next.round, local.round);
        same.assert_eq(next.query, local.query);
        same.assert_eq(next.global_query, local.global_query);
        same.assert_eq(next.coset, local.coset + AB::F::ONE);
        same.assert_eq(next.inner_tree_id, local.inner_tree_id);
        same.assert_eq(next.outer_tree_id, local.outer_tree_id);
        same.assert_eq(next.merkle_index_sample, local.merkle_index_sample);
        same.assert_eq(next.merkle_index, local.merkle_index);
        same.assert_eq(next.zi_root, local.zi_root);
        same.assert_eq(next.zi, local.zi);
        same.assert_eq(next.twiddle, local.twiddle * omega_k);
        assert_array_eq(&mut same, next.yi, local.yi);
        assert_array_eq(&mut same, next.query_digest, local.query_digest);

        let next_query = next.active * next.is_first_in_query;
        let mut transition = builder.when_transition();
        let mut advance = transition.when(next_query);
        advance.assert_eq(next.global_query, local.global_query + AB::F::ONE);
        advance.assert_zero(next.coset);

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

        // Every extension element is one scalar Merkle leaf.  Positions are
        // local to that leaf, unlike the legacy wide-vector leaf.
        for (position, value) in local.value.into_iter().enumerate() {
            self.leaf_value_bus.lookup_key(
                builder,
                NativeLeafValueMessage {
                    proof_idx: AB::Expr::ZERO,
                    tree_id: local.inner_tree_id.into(),
                    index: local.coset.into(),
                    position: AB::Expr::from_usize(position),
                    value: value.into(),
                },
                local.active,
            );
        }
        self.merkle_root_bus.receive(
            builder,
            NativeMerkleRootMessage {
                proof_idx: AB::Expr::ZERO,
                tree_id: local.inner_tree_id.into(),
                depth: AB::Expr::from_usize(self.k),
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

#[derive(Debug)]
pub struct FiniteWarpV3TwoCosetTerminalWhirOpenedTrace {
    pub matrix: RowMajorMatrix<F>,
    pub leaves: Vec<NativeTerminalOwnedLeafHashInput>,
    pub inner_merkle: Vec<(u32, BinaryMerkleMultiproofRecord<Digest>)>,
}

/// Generate scalar opening rows and the exact depth-`k` inner Merkle records.
/// Every proof-derived vector and transcript position is checked before use.
#[allow(clippy::too_many_arguments)]
pub fn generate_finite_warp_v3_two_coset_terminal_whir_opened_trace<H>(
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
    hasher: &H,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    inner_tree_id_offset: usize,
    outer_tree_id_offset: usize,
    required_height: Option<usize>,
) -> Result<FiniteWarpV3TwoCosetTerminalWhirOpenedTrace, FiniteWarpV3TwoCosetOpeningError>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    let k = profile.whir_k();
    if k != FINITE_WARP_V3_TWO_COSET_K
        || profile.alpha_len() != profile.log_message_len() + 1
        || profile.log_message_len() > F::TWO_ADICITY
        || verification.rounds.is_empty()
    {
        return Err(FiniteWarpV3TwoCosetOpeningError::UnsupportedProfile);
    }
    let coset_size = 1usize
        .checked_shl(k as u32)
        .ok_or(FiniteWarpV3TwoCosetOpeningError::UnsupportedProfile)?;
    let query_count = verification.rounds.iter().try_fold(0usize, |total, round| {
        total.checked_add(round.query_indices.len())
    });
    let query_count = query_count.ok_or(FiniteWarpV3TwoCosetOpeningError::VerificationShape)?;
    let valid_rows = query_count
        .checked_mul(coset_size)
        .ok_or(FiniteWarpV3TwoCosetOpeningError::TraceHeight)?;
    if valid_rows == 0 {
        return Err(FiniteWarpV3TwoCosetOpeningError::VerificationShape);
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FiniteWarpV3TwoCosetOpeningError::TraceHeight);
    }
    let width = NativeTerminalWhirOpenedCols::<F>::width();
    let cells = height
        .checked_mul(width)
        .ok_or(FiniteWarpV3TwoCosetOpeningError::TraceHeight)?;
    let mut values = F::zero_vec(cells);
    let mut leaves = Vec::with_capacity(valid_rows);
    let mut inner_merkle = Vec::with_capacity(query_count);
    let mut row_index = 0usize;
    let mut global_query = 0usize;
    let omega_k = F::two_adic_generator(k);

    for (round_index, round) in verification.rounds.iter().enumerate() {
        let expected_log = profile
            .alpha_len()
            .checked_sub(round_index)
            .ok_or(FiniteWarpV3TwoCosetOpeningError::UnsupportedProfile)?;
        let query_bits = expected_log
            .checked_sub(k)
            .ok_or(FiniteWarpV3TwoCosetOpeningError::UnsupportedProfile)?;
        let count = round.query_indices.len();
        if query_bits == 0
            || query_bits >= u32::BITS as usize
            || round.round as usize != round_index
            || round.log_rs_domain_size as usize != expected_log
            || round.alphas.len() != k
            || count == 0
            || round.query_roots.len() != count
            || round.folded_values.len() != count
            || round.opened_rows.len() != count
            || round.query_digests.len() != count
        {
            return Err(FiniteWarpV3TwoCosetOpeningError::VerificationShape);
        }
        let query_tidx = round
            .transcript_span
            .end
            .operations
            .checked_sub(D_EF + count)
            .ok_or(FiniteWarpV3TwoCosetOpeningError::Transcript)?;
        let mask = (1u32 << query_bits) - 1;

        for query in 0..count {
            let merkle_index = round.query_indices[query];
            let tidx = query_tidx
                .checked_add(query)
                .ok_or(FiniteWarpV3TwoCosetOpeningError::Transcript)?;
            let sample = *transcript
                .values()
                .get(tidx)
                .ok_or(FiniteWarpV3TwoCosetOpeningError::Transcript)?;
            if transcript.samples().get(tidx) != Some(&true) {
                return Err(FiniteWarpV3TwoCosetOpeningError::Transcript);
            }
            if sample.as_canonical_u32() & mask != merkle_index {
                return Err(FiniteWarpV3TwoCosetOpeningError::QueryIndex);
            }
            let raw_root = terminal_whir_query_root::<F>(
                TerminalWhirLayout::ScalarCoefficientTwoCoset,
                round_index == 0,
                merkle_index as usize,
                expected_log,
                k,
            )
            .map_err(|_| FiniteWarpV3TwoCosetOpeningError::QueryRoot)?;
            let zi = round.query_roots[query];
            if raw_root.exp_power_of_2(k) != zi {
                return Err(FiniteWarpV3TwoCosetOpeningError::QueryRoot);
            }
            let opened = &round.opened_rows[query];
            if opened.len() != coset_size || opened.iter().any(|row| row.len() != 1) {
                return Err(FiniteWarpV3TwoCosetOpeningError::OpeningShape);
            }
            let inner_tree_id = u32::try_from(
                inner_tree_id_offset
                    .checked_add(global_query)
                    .ok_or(FiniteWarpV3TwoCosetOpeningError::TreeId)?,
            )
            .map_err(|_| FiniteWarpV3TwoCosetOpeningError::TreeId)?;
            let outer_tree_id = u32::try_from(
                outer_tree_id_offset
                    .checked_add(round_index)
                    .ok_or(FiniteWarpV3TwoCosetOpeningError::TreeId)?,
            )
            .map_err(|_| FiniteWarpV3TwoCosetOpeningError::TreeId)?;
            let leaf_digests = opened
                .iter()
                .map(|row| hasher.hash_slice(row[0].as_basis_coefficients_slice()))
                .collect::<Vec<_>>();
            let leaf_indices = (0..coset_size).collect::<Vec<_>>();
            let inner = verify_binary_merkle_multiproof_recorded(
                hasher,
                round.query_digests[query],
                &leaf_indices,
                &leaf_digests,
                k,
                &BinaryMerkleMultiProof::default(),
            )
            .map_err(|_| FiniteWarpV3TwoCosetOpeningError::MerkleRoot)?;
            inner_merkle.push((inner_tree_id, inner));

            let mut twiddle = F::ONE;
            for (coset, opened_row) in opened.iter().enumerate() {
                let value = opened_row[0];
                let limbs = value.as_basis_coefficients_slice().to_vec();
                leaves.push(NativeTerminalOwnedLeafHashInput {
                    tree_id: inner_tree_id,
                    leaf_index: u32::try_from(coset)
                        .map_err(|_| FiniteWarpV3TwoCosetOpeningError::TreeId)?,
                    values: limbs,
                    lookup_counts: vec![1; D_EF],
                });
                let row = values
                    .get_mut(row_index * width..(row_index + 1) * width)
                    .ok_or(FiniteWarpV3TwoCosetOpeningError::TraceHeight)?;
                let cols: &mut NativeTerminalWhirOpenedCols<F> = row.borrow_mut();
                cols.active = F::ONE;
                cols.round = F::from_usize(round_index);
                cols.query = F::from_usize(query);
                cols.global_query = F::from_usize(global_query);
                cols.coset = F::from_usize(coset);
                cols.is_first = F::from_bool(row_index == 0);
                cols.is_first_in_round = F::from_bool(query == 0 && coset == 0);
                cols.is_first_in_query = F::from_bool(coset == 0);
                cols.inner_tree_id = F::from_u32(inner_tree_id);
                cols.outer_tree_id = F::from_u32(outer_tree_id);
                cols.merkle_index_sample = sample;
                cols.merkle_index = F::from_u32(merkle_index);
                cols.zi_root = raw_root;
                cols.zi = zi;
                copy_ext(&mut cols.yi, round.folded_values[query]);
                cols.twiddle = twiddle;
                copy_ext(&mut cols.value, value);
                cols.query_digest = round.query_digests[query];
                twiddle *= omega_k;
                row_index += 1;
            }
            global_query += 1;
        }
    }
    if row_index != valid_rows || inner_merkle.len() != query_count {
        return Err(FiniteWarpV3TwoCosetOpeningError::VerificationShape);
    }
    Ok(FiniteWarpV3TwoCosetTerminalWhirOpenedTrace {
        matrix: RowMajorMatrix::new(values, width),
        leaves,
        inner_merkle,
    })
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);
const _: () = assert!(DIGEST_SIZE == 8);

#[cfg(test)]
#[path = "terminal_two_coset_opening_tests.rs"]
mod tests;

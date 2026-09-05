//! Backend-neutral accumulator-instance digest AIR primitives.
//!
//! These helpers authenticate WARP accumulator instances. They are shared by
//! transition verification and terminal Decide, and are independent of the
//! application PESAT relation.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{
        Poseidon2CompressBus, Poseidon2CompressMessage, Poseidon2PermuteBus,
        Poseidon2PermuteMessage,
    },
    native_warp::{
        NativeAccumulatorAlgebraicDigestBus, NativeAccumulatorAlgebraicDigestMessage,
        NativeAccumulatorDigestElementBus, NativeAccumulatorDigestElementMessage,
    },
    utils::poseidon2_hash_slice_with_states,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    native_warp::{native_accumulator_instance_digest_preimage, NATIVE_ACCUMULATOR_INSTANCE_TAG},
    warp_pesat::AccumulatorInstance,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE,
    D_EF, EF, F,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

const HASH_RATE: usize = DIGEST_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeAccumulatorBindingMode {
    Prior,
    Output,
}

/// Private lookup coordinates for `(alpha, mu, beta, eta)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativePrivateAccumulatorLayout {
    pub alpha: usize,
    pub mu: usize,
    pub beta: usize,
    pub eta: usize,
    pub width: usize,
    pub alpha_len: usize,
    pub beta_len: usize,
}

impl NativePrivateAccumulatorLayout {
    #[must_use]
    pub fn new(base: usize, alpha_len: usize, beta_len: usize) -> Self {
        assert!(alpha_len > 0 && beta_len > 0);
        let alpha = base;
        let mu = alpha + alpha_len * D_EF;
        let beta = mu + D_EF;
        let eta = beta + beta_len * D_EF;
        let width = eta + D_EF;
        Self {
            alpha,
            mu,
            beta,
            eta,
            width,
            alpha_len,
            beta_len,
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeAccumulatorValueCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub is_first: T,
    pub is_last: T,
    pub section: [T; 4],
    pub coordinate: T,
    pub remaining: T,
    pub section_last: T,
    pub section_last_inverse: T,
    pub ordinal: T,
    pub value: [T; D_EF],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeAccumulatorHashCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub is_first: T,
    pub is_last: T,
    pub chunk_index: T,
    pub elements: [T; HASH_RATE],
    pub capacity: [T; DIGEST_SIZE],
    pub permutation_output: [T; POSEIDON2_WIDTH],
    pub final_digest: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(NativeAccumulatorHashCols<u8>)]
pub struct NativeAccumulatorHashAir {
    pub state: usize,
    /// Some setup-fixed product components (notably reduced-SWIRL prior
    /// authentication) are legitimately empty for a bootstrap-only block.
    /// Existing callers keep this false and preserve the original non-empty
    /// invariant.
    pub allow_empty: bool,
    pub alpha_len: usize,
    pub beta_len: usize,
    pub digest_element_bus: NativeAccumulatorDigestElementBus,
    pub algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus,
    pub permute_bus: Poseidon2PermuteBus,
    pub compress_bus: Poseidon2CompressBus,
}

impl NativeAccumulatorHashAir {
    fn preimage_len(&self) -> usize {
        3 + (self.alpha_len + self.beta_len + 2) * D_EF
    }

    fn chunk_count(&self) -> usize {
        self.preimage_len().div_ceil(HASH_RATE)
    }
}

impl BaseAirWithPublicValues<F> for NativeAccumulatorHashAir {}
impl PartitionedBaseAir<F> for NativeAccumulatorHashAir {}
impl BaseAir<F> for NativeAccumulatorHashAir {
    fn width(&self) -> usize {
        NativeAccumulatorHashCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeAccumulatorHashAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("native accumulator hash row");
        let next_row = main.row_slice(1).expect("native accumulator hash next row");
        let local: &NativeAccumulatorHashCols<AB::Var> = (*local_row).borrow();
        let next: &NativeAccumulatorHashCols<AB::Var> = (*next_row).borrow();
        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        if !self.allow_empty {
            builder.when_first_row().assert_one(local.active);
        }
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.chunk_index);
        for value in local.capacity {
            builder.when_first_row().assert_zero(value);
        }
        builder.when(local.active * local.is_last).assert_eq(
            local.chunk_index,
            AB::Expr::from_usize(self.chunk_count() - 1),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder.when_transition().assert_eq(
            local.active - next.active,
            local.is_last * (AB::Expr::ONE - next.active),
        );
        let mut transition = builder.when_transition();
        let mut next_active = transition.when(next.active * (AB::Expr::ONE - next.is_first));
        next_active.assert_eq(next.proof_idx, local.proof_idx);
        next_active.assert_eq(next.chunk_index, local.chunk_index + AB::F::ONE);
        for limb in 0..DIGEST_SIZE {
            next_active.assert_eq(
                next.capacity[limb],
                local.permutation_output[DIGEST_SIZE + limb],
            );
        }
        let mut transition = builder.when_transition();
        let mut next_proof = transition.when(next.active * next.is_first);
        next_proof.assert_one(local.is_last);
        next_proof.assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        next_proof.assert_zero(next.chunk_index);
        for value in next.capacity {
            next_proof.assert_zero(value);
        }
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        let final_remainder = {
            let remainder = self.preimage_len() % HASH_RATE;
            if remainder == 0 {
                HASH_RATE
            } else {
                remainder
            }
        };
        let metadata = [
            F::from_u64(NATIVE_ACCUMULATOR_INSTANCE_TAG),
            F::from_usize(self.alpha_len),
            F::from_usize(self.beta_len),
        ];
        for offset in 0..HASH_RATE {
            if offset < metadata.len() {
                builder
                    .when(local.active * local.is_first)
                    .assert_eq(local.elements[offset], metadata[offset]);
            }
            if offset >= final_remainder {
                builder
                    .when_transition()
                    .when(next.active * next.is_last)
                    .assert_eq(next.elements[offset], local.permutation_output[offset]);
            }
            let metadata_here = if offset < metadata.len() {
                local.is_first.into()
            } else {
                AB::Expr::ZERO
            };
            let padding_here = if offset >= final_remainder {
                local.is_last.into()
            } else {
                AB::Expr::ZERO
            };
            self.digest_element_bus.receive(
                builder,
                NativeAccumulatorDigestElementMessage {
                    proof_idx: local.proof_idx.into(),
                    state: AB::Expr::from_usize(self.state),
                    index: local.chunk_index * AB::Expr::from_usize(HASH_RATE)
                        + AB::Expr::from_usize(offset),
                    value: local.elements[offset].into(),
                },
                local.active * (AB::Expr::ONE - metadata_here) * (AB::Expr::ONE - padding_here),
            );
        }

        let poseidon_input = core::array::from_fn(|index| {
            if index < HASH_RATE {
                local.elements[index].into()
            } else {
                local.capacity[index - HASH_RATE].into()
            }
        });
        self.permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: poseidon_input.clone(),
                output: local.permutation_output.map(Into::into),
            },
            local.active * (AB::Expr::ONE - local.is_last),
        );
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: poseidon_input,
                output: local.final_digest.map(Into::into),
            },
            local.active * local.is_last,
        );
        self.algebraic_digest_bus.send(
            builder,
            NativeAccumulatorAlgebraicDigestMessage {
                proof_idx: local.proof_idx.into(),
                state: AB::Expr::from_usize(self.state),
                digest: local.final_digest.map(Into::into),
            },
            local.active * local.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeAccumulatorRootDigestCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub root: [T; DIGEST_SIZE],
    pub algebraic_digest: [T; DIGEST_SIZE],
    pub instance_digest: [T; DIGEST_SIZE],
}

pub struct NativeAccumulatorDigestTraces {
    pub values: RowMajorMatrix<F>,
    pub hash: RowMajorMatrix<F>,
    pub root: RowMajorMatrix<F>,
    pub poseidon2_permute_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub instance_digest: Digest,
}

#[must_use]
pub fn generate_native_accumulator_digest_traces(
    proof_idx: usize,
    instance: &AccumulatorInstance<EF, Digest>,
    layout: &NativePrivateAccumulatorLayout,
) -> Option<NativeAccumulatorDigestTraces> {
    if instance.alpha.len() != layout.alpha_len || instance.beta.len() != layout.beta_len {
        return None;
    }
    let extension_values = instance
        .alpha
        .iter()
        .copied()
        .chain(core::iter::once(instance.mu))
        .chain(instance.beta.iter().copied())
        .chain(core::iter::once(instance.eta))
        .collect::<Vec<_>>();
    let valid_value_rows = extension_values.len();
    let value_height = valid_value_rows.next_power_of_two();
    let value_width = NativeAccumulatorValueCols::<F>::width();
    let mut value_trace = vec![F::ZERO; value_height * value_width];
    for (ordinal, value) in extension_values.iter().enumerate() {
        let (section, coordinate, section_len) = if ordinal < layout.alpha_len {
            (0, ordinal, layout.alpha_len)
        } else if ordinal == layout.alpha_len {
            (1, 0, 1)
        } else if ordinal < layout.alpha_len + 1 + layout.beta_len {
            (2, ordinal - layout.alpha_len - 1, layout.beta_len)
        } else {
            (3, 0, 1)
        };
        let row = &mut value_trace[ordinal * value_width..(ordinal + 1) * value_width];
        let cols: &mut NativeAccumulatorValueCols<F> = row.borrow_mut();
        let remaining = section_len - coordinate;
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_first = F::from_bool(ordinal == 0);
        cols.is_last = F::from_bool(ordinal + 1 == valid_value_rows);
        cols.section[section] = F::ONE;
        cols.coordinate = F::from_usize(coordinate);
        cols.remaining = F::from_usize(remaining);
        cols.section_last = F::from_bool(remaining == 1);
        cols.section_last_inverse = if remaining == 1 {
            F::ZERO
        } else {
            F::from_usize(remaining - 1).inverse()
        };
        cols.ordinal = F::from_usize(ordinal);
        cols.value
            .copy_from_slice(value.as_basis_coefficients_slice());
    }

    let preimage = native_accumulator_instance_digest_preimage::<NativeSC>(instance);
    let (algebraic_digest, pre_states, post_states) = poseidon2_hash_slice_with_states(&preimage);
    let chunk_count = pre_states.len();
    let hash_height = chunk_count.next_power_of_two();
    let hash_width = NativeAccumulatorHashCols::<F>::width();
    let mut hash_trace = vec![F::ZERO; hash_height * hash_width];
    for chunk in 0..chunk_count {
        let row = &mut hash_trace[chunk * hash_width..(chunk + 1) * hash_width];
        let cols: &mut NativeAccumulatorHashCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_first = F::from_bool(chunk == 0);
        cols.is_last = F::from_bool(chunk + 1 == chunk_count);
        cols.chunk_index = F::from_usize(chunk);
        cols.elements
            .copy_from_slice(&pre_states[chunk][..HASH_RATE]);
        cols.capacity
            .copy_from_slice(&pre_states[chunk][HASH_RATE..]);
        if chunk + 1 == chunk_count {
            cols.final_digest = algebraic_digest;
        } else {
            cols.permutation_output = post_states[chunk];
        }
    }

    let instance_digest = poseidon2_compress_with_capacity(instance.rt, algebraic_digest).0;
    let root_width = NativeAccumulatorRootDigestCols::<F>::width();
    let mut root_trace = vec![F::ZERO; root_width];
    let root_cols: &mut NativeAccumulatorRootDigestCols<F> = root_trace.as_mut_slice().borrow_mut();
    root_cols.active = F::ONE;
    root_cols.proof_idx = F::from_usize(proof_idx);
    root_cols.root = instance.rt;
    root_cols.algebraic_digest = algebraic_digest;
    root_cols.instance_digest = instance_digest;

    let mut poseidon2_compress_inputs = vec![pre_states[chunk_count - 1]];
    poseidon2_compress_inputs.push(core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            instance.rt[index]
        } else {
            algebraic_digest[index - DIGEST_SIZE]
        }
    }));
    Some(NativeAccumulatorDigestTraces {
        values: RowMajorMatrix::new(value_trace, value_width),
        hash: RowMajorMatrix::new(hash_trace, hash_width),
        root: RowMajorMatrix::new(root_trace, root_width),
        poseidon2_permute_inputs: pre_states[..chunk_count - 1].to_vec(),
        poseidon2_compress_inputs,
        instance_digest,
    })
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::{native_warp::native_accumulator_instance_digest, SystemParams};
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

    use super::*;

    #[test]
    fn compact_accumulator_digest_matches_host_binding() {
        let params = SystemParams::new_for_testing(10);
        let config = BabyBearPoseidon2Config::default_from_params(params);
        let instance = AccumulatorInstance {
            rt: [F::from_u32(17); DIGEST_SIZE],
            alpha: (0..5).map(|i| EF::from(F::from_usize(i + 1))).collect(),
            mu: EF::from(F::from_u32(91)),
            beta: (0..7).map(|i| EF::from(F::from_usize(i + 20))).collect(),
            eta: EF::from(F::from_u32(123)),
        };
        let layout = NativePrivateAccumulatorLayout::new(100, 5, 7);
        let traces = generate_native_accumulator_digest_traces(0, &instance, &layout).unwrap();
        assert_eq!(
            traces.instance_digest,
            native_accumulator_instance_digest(&config, &instance)
        );
        assert_eq!(traces.values.height(), 16);
        assert_eq!(traces.hash.height(), 8);
        assert_eq!(
            traces.poseidon2_permute_inputs.len() + traces.poseidon2_compress_inputs.len(),
            9
        );
    }
}

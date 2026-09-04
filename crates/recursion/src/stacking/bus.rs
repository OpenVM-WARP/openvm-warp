use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;

use crate::{define_typed_per_proof_lookup_bus, define_typed_per_proof_permutation_bus};

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StackingModuleTidxMessage<T> {
    pub module_idx: T,
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(StackingModuleTidxBus, StackingModuleTidxMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ClaimCoefficientsMessage<T> {
    pub commit_idx: T,
    pub stacked_col_idx: T,
    pub coefficient: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(ClaimCoefficientsBus, ClaimCoefficientsMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct SumcheckClaimsMessage<T> {
    pub module_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(SumcheckClaimsBus, SumcheckClaimsMessage);

/// One flattened stacking opening produced by the ordinary stacking verifier.
///
/// `proof_idx` (inserted by the typed bus) is the ordered reduction index and
/// `opening_idx` is commitment-major. Setup-PCS authority consumes this bus
/// exactly once before publishing the value on its multi-constraint statement
/// bus; the value therefore cannot be replaced by an unconstrained adapter
/// witness.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct OrderedStackingOpeningMessage<T> {
    pub opening_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(OrderedStackingOpeningBus, OrderedStackingOpeningMessage);

/// One coordinate of the original source opening point used as `r` by the
/// ordinary stacking equations.
///
/// `proof_idx` (inserted by the typed bus) is the ordered reduction/transition
/// index. In ordered mode the setup authority sends every coordinate exactly
/// once. `EqBaseAir` consumes coordinate zero and `SumcheckRoundsAir` consumes
/// the remaining coordinates, so the values entering the stacking algebra are
/// constrained by the authority boundary rather than merely compared by host
/// trace generation.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct OrderedStackingSourcePointMessage<T> {
    pub coordinate_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(
    OrderedStackingSourcePointBus,
    OrderedStackingSourcePointMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqRandValuesLookupMessage<T> {
    pub idx: T,
    pub u: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(EqRandValuesLookupBus, EqRandValuesLookupMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqBaseMessage<T> {
    // eq_0(u, r)
    pub eq_u_r: [T; D_EF],
    // eq_0(u, r * omega)
    pub eq_u_r_omega: [T; D_EF],
    // eq_0(u, 1) * eq_0(r, \omega^{-1})
    pub eq_u_r_prod: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqBaseBus, EqBaseMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqKernelLookupMessage<T> {
    pub n: T,
    pub eq_in: [T; D_EF],
    pub k_rot_in: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(EqKernelLookupBus, EqKernelLookupMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqBitsLookupMessage<T> {
    // most significant num_bits bits of row_idx
    pub b_value: T,
    // n_{stack} - n_j
    pub num_bits: T,
    // eq_{n_{stack} - n_j}(u_{> n_j}, b_j)
    pub eval: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(EqBitsLookupBus, EqBitsLookupMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqBitsInternalMessage<T> {
    // most significant num_bits bits of row_idx without lsb
    pub b_value: T,
    // n_{stack} - n_j - 1
    pub num_bits: T,
    // eq_{n_{stack} - n_j - 1}(u_{> n_j}, b_j)
    pub eval: [T; D_EF],
    // least significant bit of the b_value of the row that receives this message
    pub child_lsb: T,
}

define_typed_per_proof_permutation_bus!(EqBitsInternalBus, EqBitsInternalMessage);

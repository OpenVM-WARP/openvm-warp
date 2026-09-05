use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::native_warp::{
    bus::{NativeClaimLayoutBus, NativeClaimLayoutMessage},
    twin::{CLAIM_SECTION_ALPHA, CLAIM_SECTION_BETA, CLAIM_SECTION_ETA, CLAIM_SECTION_MU},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeClaimValueCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub source: T,
    pub section: T,
    /// One-hot alpha, beta, mu, eta section selector.
    pub section_kind: [T; 4],
    pub coordinate: T,
    pub is_fresh: T,
    pub is_prior: T,
    pub is_dummy: T,
    pub variant: T,
    pub transcript_id: T,
    pub tidx: T,
    pub is_transcript: T,
    pub is_sample: T,
    pub is_beta_tail: T,
    pub tail_coordinate: T,
    pub value: [T; D_EF],
}

#[derive(Clone, Debug)]
pub struct NativeClaimTraceInput<'a> {
    pub proof_idx: usize,
    pub alpha: &'a [EF],
    pub beta: &'a [EF],
    pub mu: EF,
    pub eta: EF,
    pub is_fresh: bool,
    pub is_prior: bool,
    pub is_dummy: bool,
    /// `(transcript_id, tidx, is_sample)` for each alpha, beta, mu, eta row in
    /// the same flattened order emitted by the trace generator.
    pub transcript_bindings: &'a [Option<(usize, usize, bool)>],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeClaimLayoutCols<T> {
    pub active: T,
    pub section: T,
    pub coordinate: T,
    pub is_beta_tail: T,
    pub tail_coordinate: T,
    pub lookup_count: T,
}

#[derive(ColumnsAir)]
#[columns_via(NativeClaimLayoutCols<u8>)]
pub struct NativeClaimLayoutAir {
    pub layout_bus: NativeClaimLayoutBus,
}

impl BaseAirWithPublicValues<F> for NativeClaimLayoutAir {}
impl PartitionedBaseAir<F> for NativeClaimLayoutAir {}
impl<F> BaseAir<F> for NativeClaimLayoutAir {
    fn width(&self) -> usize {
        NativeClaimLayoutCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeClaimLayoutAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native claim layout row");
        let local: &NativeClaimLayoutCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_beta_tail);
        self.layout_bus.add_key_with_lookups(
            builder,
            NativeClaimLayoutMessage {
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                is_beta_tail: local.is_beta_tail.into(),
                tail_coordinate: local.tail_coordinate.into(),
            },
            local.lookup_count,
        );
    }
}

pub fn generate_native_claim_value_trace(
    claims: &[NativeClaimTraceInput<'_>],
    beta_prefix_len: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if claims.is_empty() {
        return None;
    }
    let rows_per_claim = claims[0].alpha.len() + claims[0].beta.len() + 2;
    if claims.iter().any(|claim| {
        claim.alpha.len() != claims[0].alpha.len()
            || claim.beta.len() != claims[0].beta.len()
            || usize::from(claim.is_fresh)
                + usize::from(claim.is_prior)
                + usize::from(claim.is_dummy)
                != 1
            || claim.transcript_bindings.len() != rows_per_claim
    }) {
        return None;
    }
    let valid_rows = claims.len() * rows_per_claim;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeClaimValueCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut row_index = 0usize;
    for (source, claim) in claims.iter().enumerate() {
        let mut binding_index = 0usize;
        for (section, values) in [
            (CLAIM_SECTION_ALPHA, claim.alpha),
            (CLAIM_SECTION_BETA, claim.beta),
            (CLAIM_SECTION_MU, core::slice::from_ref(&claim.mu)),
            (CLAIM_SECTION_ETA, core::slice::from_ref(&claim.eta)),
        ] {
            for (coordinate, value) in values.iter().enumerate() {
                let cols: &mut NativeClaimValueCols<F> =
                    trace[row_index * width..(row_index + 1) * width].borrow_mut();
                cols.active = F::ONE;
                cols.proof_idx = F::from_usize(claim.proof_idx);
                cols.source = F::from_usize(source);
                cols.section = F::from_usize(section);
                cols.section_kind[section] = F::ONE;
                cols.coordinate = F::from_usize(coordinate);
                cols.is_fresh = F::from_bool(claim.is_fresh);
                cols.is_prior = F::from_bool(claim.is_prior);
                cols.is_dummy = F::from_bool(claim.is_dummy);
                cols.variant = F::from_usize(
                    claims.iter().filter(|claim| claim.is_fresh).count()
                        + usize::from(claims.iter().any(|claim| claim.is_prior))
                            * (claims.len() + 1),
                );
                if let Some((transcript_id, tidx, is_sample)) =
                    claim.transcript_bindings[binding_index]
                {
                    cols.transcript_id = F::from_usize(transcript_id);
                    cols.tidx = F::from_usize(tidx);
                    cols.is_transcript = F::ONE;
                    cols.is_sample = F::from_bool(is_sample);
                }
                cols.value
                    .copy_from_slice(value.as_basis_coefficients_slice());
                if section == CLAIM_SECTION_BETA && coordinate >= beta_prefix_len {
                    cols.is_beta_tail = F::ONE;
                    cols.tail_coordinate = F::from_usize(coordinate - beta_prefix_len);
                }
                binding_index += 1;
                row_index += 1;
            }
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

pub fn generate_native_claim_layout_trace(
    alpha_len: usize,
    beta_len: usize,
    beta_prefix_len: usize,
    claim_count: usize,
) -> Option<RowMajorMatrix<F>> {
    if alpha_len == 0 || beta_prefix_len > beta_len || claim_count == 0 {
        return None;
    }
    let rows = alpha_len + beta_len + 2;
    let height = rows.next_power_of_two();
    let width = NativeClaimLayoutCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut row_index = 0usize;
    for (section, len) in [
        (CLAIM_SECTION_ALPHA, alpha_len),
        (CLAIM_SECTION_BETA, beta_len),
        (CLAIM_SECTION_MU, 1),
        (CLAIM_SECTION_ETA, 1),
    ] {
        for coordinate in 0..len {
            let cols: &mut NativeClaimLayoutCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.section = F::from_usize(section);
            cols.coordinate = F::from_usize(coordinate);
            if section == CLAIM_SECTION_BETA && coordinate >= beta_prefix_len {
                cols.is_beta_tail = F::ONE;
                cols.tail_coordinate = F::from_usize(coordinate - beta_prefix_len);
            }
            cols.lookup_count = F::from_usize(claim_count);
            row_index += 1;
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

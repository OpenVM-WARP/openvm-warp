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

use crate::{
    bus::TranscriptBus,
    native_warp::{
        bus::{
            NativeClaimLayoutBus, NativeClaimLayoutMessage, NativeClaimValueBus,
            NativeClaimValueMessage, NativeExplicitValueBus, NativeExplicitValueMessage,
            NativeInputSlotLayoutBus, NativeInputSlotLayoutMessage, NativePcdStateReadBus,
            NativePcdStateReadMessage,
        },
        twin::{CLAIM_SECTION_ALPHA, CLAIM_SECTION_BETA, CLAIM_SECTION_ETA, CLAIM_SECTION_MU},
    },
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

#[derive(ColumnsAir)]
#[columns_via(NativeClaimValueCols<u8>)]
pub struct NativeClaimValueAir {
    pub claim_bus: NativeClaimValueBus,
    pub explicit_bus: NativeExplicitValueBus,
    pub transcript_bus: TranscriptBus,
    pub layout_bus: NativeClaimLayoutBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub state_read_bus: NativePcdStateReadBus,
    pub pvs_alpha_offset: usize,
    pub pvs_beta_offset: usize,
    pub pvs_mu_offset: usize,
    pub pvs_eta_offset: usize,
    /// Whether all claim rows feed the in-circuit twin algebra. Certified
    /// replay disables this consumer; only the independent fresh-input digest
    /// remains inside the history relation.
    pub bind_vacc_algebra: bool,
    /// Number of consumers for fresh claim rows. Ordinary VACC uses one;
    /// the bounded certificate adds one public-input binding consumer.
    pub fresh_claim_consumer_count: usize,
    /// Legacy stacked-opening relations use a zero fresh `alpha`.  Direct-AIR
    /// protocol v19 instead consumes the systematic RS lift `(r, 0^b)`, so
    /// only `eta` is universally zero.  Keep this fixed in the AIR key rather
    /// than accepting a witness selector between the two protocols.
    pub fresh_alpha_is_zero: bool,
}

impl BaseAirWithPublicValues<F> for NativeClaimValueAir {}
impl PartitionedBaseAir<F> for NativeClaimValueAir {}
impl<F> BaseAir<F> for NativeClaimValueAir {
    fn width(&self) -> usize {
        NativeClaimValueCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeClaimValueAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native claim value row");
        let local: &NativeClaimValueCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_fresh);
        builder.assert_bool(local.is_prior);
        builder.assert_bool(local.is_dummy);
        builder.assert_bool(local.is_transcript);
        builder.assert_bool(local.is_sample);
        builder.assert_bool(local.is_beta_tail);
        builder
            .when(local.active * local.is_transcript)
            .assert_eq(local.proof_idx, local.transcript_id);
        for flag in local.section_kind {
            builder.assert_bool(flag);
        }
        builder
            .when(local.active)
            .assert_one(local.is_fresh + local.is_prior + local.is_dummy);
        builder.assert_eq(
            local.active,
            local
                .section_kind
                .into_iter()
                .map(AB::Expr::from)
                .sum::<AB::Expr>(),
        );
        let [is_alpha, is_beta, is_mu, is_eta] = local.section_kind.map(AB::Expr::from);
        builder.assert_eq(
            local.section,
            is_beta.clone()
                + is_mu.clone() * AB::Expr::from_usize(CLAIM_SECTION_MU)
                + is_eta.clone() * AB::Expr::from_usize(CLAIM_SECTION_ETA),
        );
        self.slot_bus.lookup_key(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: [
                    local.is_fresh.into(),
                    local.is_prior.into(),
                    local.is_dummy.into(),
                ],
            },
            local.active,
        );
        let vector_selector = is_alpha.clone() + is_beta.clone();
        for limb in 0..D_EF {
            let pvs_coordinate = is_alpha.clone() * AB::Expr::from_usize(self.pvs_alpha_offset)
                + is_beta.clone() * AB::Expr::from_usize(self.pvs_beta_offset)
                + is_mu.clone() * AB::Expr::from_usize(self.pvs_mu_offset)
                + is_eta.clone() * AB::Expr::from_usize(self.pvs_eta_offset)
                + vector_selector.clone() * local.coordinate * AB::Expr::from_usize(D_EF)
                + AB::Expr::from_usize(limb);
            self.state_read_bus.lookup_key(
                builder,
                NativePcdStateReadMessage {
                    state: AB::Expr::ZERO,
                    coordinate: pvs_coordinate,
                    value: local.value[limb].into(),
                },
                local.active * local.is_prior,
            );
        }
        for limb in local.value {
            builder
                .when(local.active * local.is_dummy)
                .assert_zero(limb);
            builder
                .when(
                    local.is_fresh
                        * (is_eta.clone()
                            + is_alpha.clone() * AB::Expr::from_bool(self.fresh_alpha_is_zero)),
                )
                .assert_zero(limb);
        }
        let vacc_consumer = AB::Expr::from_bool(self.bind_vacc_algebra);
        self.claim_bus.send(
            builder,
            NativeClaimValueMessage {
                proof_idx: local.proof_idx.into(),
                source: local.source.into(),
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active
                * (vacc_consumer.clone()
                    + local.is_fresh
                        * (AB::Expr::from_usize(self.fresh_claim_consumer_count) - vacc_consumer)),
        );
        self.layout_bus.lookup_key(
            builder,
            NativeClaimLayoutMessage {
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                is_beta_tail: local.is_beta_tail.into(),
                tail_coordinate: local.tail_coordinate.into(),
            },
            local.active,
        );
        self.explicit_bus.lookup_key(
            builder,
            NativeExplicitValueMessage {
                source: local.source.into(),
                coordinate: local.tail_coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active * local.is_beta_tail * local.is_fresh,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.transcript_id,
            local.tidx,
            local.value,
            local.active * local.is_transcript * (AB::Expr::ONE - local.is_sample),
        );
        self.transcript_bus.sample_ext(
            builder,
            local.transcript_id,
            local.tidx,
            local.value,
            local.active * local.is_transcript * local.is_sample,
        );
    }
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

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::air_builders::debug::check_constraints;
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as NativeSC;

    use super::*;

    #[test]
    fn generic_claim_rejects_two_proof_transcript_cross_splice() {
        let alpha = [EF::from_u32(3)];
        let beta = [EF::from_u32(5)];
        let first_bindings = [
            Some((0, 10, false)),
            Some((0, 14, true)),
            Some((0, 18, false)),
            Some((0, 22, false)),
        ];
        let second_bindings = [
            Some((1, 10, false)),
            Some((1, 14, true)),
            Some((1, 18, false)),
            Some((1, 22, false)),
        ];
        let claims = [
            NativeClaimTraceInput {
                proof_idx: 0,
                alpha: &alpha,
                beta: &beta,
                mu: EF::from_u32(7),
                eta: EF::ZERO,
                is_fresh: true,
                is_prior: false,
                is_dummy: false,
                transcript_bindings: &first_bindings,
            },
            NativeClaimTraceInput {
                proof_idx: 1,
                alpha: &alpha,
                beta: &beta,
                mu: EF::from_u32(11),
                eta: EF::ZERO,
                is_fresh: true,
                is_prior: false,
                is_dummy: false,
                transcript_bindings: &second_bindings,
            },
        ];
        let mut trace = generate_native_claim_value_trace(&claims, 1, None).unwrap();
        let air = NativeClaimValueAir {
            claim_bus: NativeClaimValueBus::new(0),
            explicit_bus: NativeExplicitValueBus::new(1),
            transcript_bus: TranscriptBus::new(2),
            layout_bus: NativeClaimLayoutBus::new(3),
            slot_bus: NativeInputSlotLayoutBus::new(4),
            state_read_bus: NativePcdStateReadBus::new(5),
            pvs_alpha_offset: 0,
            pvs_beta_offset: D_EF,
            pvs_mu_offset: 2 * D_EF,
            pvs_eta_offset: 3 * D_EF,
            bind_vacc_algebra: true,
            fresh_claim_consumer_count: 1,
            fresh_alpha_is_zero: false,
        };
        check_constraints::<_, NativeSC>(
            &air,
            "NativeClaimValueAir",
            &None,
            &[trace.as_view()],
            &[],
        );

        let rows_per_claim = alpha.len() + beta.len() + 2;
        let width = NativeClaimValueCols::<F>::width();
        let second_first = &mut trace.values[rows_per_claim * width..(rows_per_claim + 1) * width];
        let cols: &mut NativeClaimValueCols<F> = second_first.borrow_mut();
        assert_eq!(cols.proof_idx, F::ONE);
        assert_eq!(cols.transcript_id, F::ONE);
        cols.transcript_id = F::ZERO;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, NativeSC>(
                &air,
                "NativeClaimValueAir",
                &None,
                &[trace.as_view()],
                &[],
            );
        }))
        .is_err());
    }
}

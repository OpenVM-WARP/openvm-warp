use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::{
        batching::{OPENING_SECTION_POINT, OPENING_SECTION_TARGET},
        bus::{
            NativeAuthenticatedShiftBus, NativeAuthenticatedShiftMessage, NativeEqResultBus,
            NativeEqResultMessage, NativeInputSlotLayoutBus, NativeInputSlotLayoutMessage,
            NativeOpeningClaimBus, NativeOpeningClaimMessage,
        },
        ext::{ext_field_add, ext_field_multiply},
    },
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage, RightShiftBus, RightShiftMessage},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeShiftMergeCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub shift: T,
    pub source: T,
    pub is_first_source: T,
    pub is_last_source: T,
    pub is_first_shift: T,
    pub variant: T,
    pub source_kind: [T; 3],
    pub gamma_weight: [T; D_EF],
    pub authenticated_value: [T; D_EF],
    pub accumulator_before: [T; D_EF],
    pub accumulator_after: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeShiftMergeCols<u8>)]
pub struct NativeShiftMergeAir {
    pub authenticated_bus: NativeAuthenticatedShiftBus,
    pub eq_bus: NativeEqResultBus,
    pub opening_bus: NativeOpeningClaimBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub input_arity: usize,
    pub gamma_eq_group_offset: usize,
    pub opening_claim_offset: usize,
}

impl BaseAirWithPublicValues<F> for NativeShiftMergeAir {}
impl PartitionedBaseAir<F> for NativeShiftMergeAir {}
impl<F> BaseAir<F> for NativeShiftMergeAir {
    fn width(&self) -> usize {
        NativeShiftMergeCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeShiftMergeAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("native shift merge row"),
            main.row_slice(1).expect("native shift merge next row"),
        );
        let local: &NativeShiftMergeCols<AB::Var> = (*local).borrow();
        let next: &NativeShiftMergeCols<AB::Var> = (*next).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_first_source);
        builder.assert_bool(local.is_last_source);
        builder.assert_bool(local.is_first_shift);
        builder
            .when(local.active * local.is_first_shift)
            .assert_one(local.is_first_source);
        builder
            .when(local.active * local.is_first_shift)
            .assert_zero(local.shift);
        for flag in local.source_kind {
            builder.assert_bool(flag);
        }
        builder.when(local.active).assert_one(
            local
                .source_kind
                .into_iter()
                .map(AB::Expr::from)
                .sum::<AB::Expr>(),
        );
        self.slot_bus.lookup_key(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: local.source_kind.map(Into::into),
            },
            local.active,
        );
        builder.when_first_row().assert_one(local.active);
        builder
            .when(local.active * local.is_first_source)
            .assert_zero(local.source);
        builder
            .when(local.active * local.is_last_source)
            .assert_eq(local.source, AB::Expr::from_usize(self.input_arity - 1));
        self.authenticated_bus.receive(
            builder,
            NativeAuthenticatedShiftMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                source: local.source.into(),
                value: local.authenticated_value.map(Into::into),
            },
            local.active * (local.source_kind[0] + local.source_kind[1]),
        );
        for limb in local.authenticated_value {
            builder
                .when(local.active * local.source_kind[2])
                .assert_zero(limb);
        }
        self.eq_bus.lookup_key(
            builder,
            NativeEqResultMessage {
                proof_idx: local.proof_idx.into(),
                group: AB::Expr::from_usize(self.gamma_eq_group_offset) + local.source,
                value: local.gamma_weight.map(Into::into),
            },
            local.active,
        );
        let zero = [AB::Expr::ZERO; D_EF];
        assert_array_eq(
            &mut builder.when(local.active * local.is_first_source),
            local.accumulator_before,
            zero,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.accumulator_after,
            ext_field_add::<AB::Expr>(
                local.accumulator_before,
                ext_field_multiply::<AB::Expr>(local.gamma_weight, local.authenticated_value),
            ),
        );
        let same_shift = next.active * (AB::Expr::ONE - next.is_first_source);
        let mut transition = builder.when_transition();
        let mut when_same = transition.when(same_shift);
        when_same.assert_zero(local.is_last_source);
        when_same.assert_eq(next.proof_idx, local.proof_idx);
        when_same.assert_zero(next.is_first_shift);
        when_same.assert_eq(next.shift, local.shift);
        when_same.assert_eq(next.source, local.source + AB::F::ONE);
        assert_array_eq(
            &mut when_same,
            next.accumulator_before,
            local.accumulator_after,
        );
        let next_shift = next.active * next.is_first_source;
        let mut transition = builder.when_transition();
        let mut when_next = transition.when(next_shift);
        when_next.assert_one(local.is_last_source);
        when_next
            .when(next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        when_next
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx);
        when_next
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.shift, local.shift + AB::F::ONE);
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: AB::Expr::from_usize(self.opening_claim_offset) + local.shift,
                section: AB::Expr::from_usize(OPENING_SECTION_TARGET),
                coordinate: AB::Expr::ZERO,
                value: local.accumulator_after.map(Into::into),
            },
            local.active * local.is_last_source,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeShiftPointCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub shift: T,
    pub coordinate: T,
    pub bit: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeShiftPointCols<u8>)]
pub struct NativeShiftPointAir {
    pub opening_bus: NativeOpeningClaimBus,
    pub opening_claim_offset: usize,
}

impl BaseAirWithPublicValues<F> for NativeShiftPointAir {}
impl PartitionedBaseAir<F> for NativeShiftPointAir {}
impl<F> BaseAir<F> for NativeShiftPointAir {
    fn width(&self) -> usize {
        NativeShiftPointCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeShiftPointAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native shift point row");
        let local: &NativeShiftPointCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when(local.active).assert_bool(local.bit);
        builder
            .when(local.active)
            .assert_eq(local.value[0], local.bit);
        for limb in &local.value[1..] {
            builder.when(local.active).assert_zero(*limb);
        }
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: AB::Expr::from_usize(self.opening_claim_offset) + local.shift,
                section: AB::Expr::from_usize(OPENING_SECTION_POINT),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_native_shift_merge_trace(
    proof_idx: usize,
    weights: &[EF],
    shift_answers: &[Vec<EF>],
    fresh_count: usize,
    prior_present: bool,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if weights.is_empty()
        || fresh_count + usize::from(prior_present) > weights.len()
        || shift_answers.is_empty()
        || shift_answers
            .iter()
            .any(|answers| answers.len() != weights.len())
    {
        return None;
    }
    let valid_rows = weights.len() * shift_answers.len();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeShiftMergeCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (shift, answers) in shift_answers.iter().enumerate() {
        let mut accumulator = EF::ZERO;
        for source in 0..weights.len() {
            let before = accumulator;
            accumulator += weights[source] * answers[source];
            let row_index = shift * weights.len() + source;
            let cols: &mut NativeShiftMergeCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.shift = F::from_usize(shift);
            cols.source = F::from_usize(source);
            cols.is_first_source = F::from_bool(source == 0);
            cols.is_last_source = F::from_bool(source + 1 == weights.len());
            cols.is_first_shift = F::from_bool(shift == 0 && source == 0);
            cols.variant =
                F::from_usize(fresh_count + usize::from(prior_present) * (weights.len() + 1));
            cols.source_kind[if source < fresh_count {
                0
            } else if prior_present && source == fresh_count {
                1
            } else {
                2
            }] = F::ONE;
            for (target, value) in [
                (&mut cols.gamma_weight, weights[source]),
                (&mut cols.authenticated_value, answers[source]),
                (&mut cols.accumulator_before, before),
                (&mut cols.accumulator_after, accumulator),
            ] {
                target.copy_from_slice(value.as_basis_coefficients_slice());
            }
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

pub fn generate_native_shift_point_trace(
    proof_idx: usize,
    points: &[Vec<EF>],
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let dimension = points.first()?.len();
    if dimension == 0 || points.iter().any(|point| point.len() != dimension) {
        return None;
    }
    let valid_rows = points.len() * dimension;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeShiftPointCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (shift, point) in points.iter().enumerate() {
        for (coordinate, &value) in point.iter().enumerate() {
            let basis = value.as_basis_coefficients_slice();
            if basis[1..].iter().any(|&limb| limb != F::ZERO)
                || (basis[0] != F::ZERO && basis[0] != F::ONE)
            {
                return None;
            }
            let row_index = shift * dimension + coordinate;
            let cols: &mut NativeShiftPointCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.shift = F::from_usize(shift);
            cols.coordinate = F::from_usize(coordinate);
            cols.bit = basis[0];
            cols.value.copy_from_slice(basis);
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeShiftScheduleCols<T> {
    pub active: T,
    pub shift: T,
    pub coordinate: T,
    pub is_first: T,
    pub is_last: T,
    pub is_first_shift: T,
    pub proof_idx: T,
    pub tidx: T,
    pub sample: T,
    pub accepted_inverse: T,
    pub quotient: T,
    pub index: T,
    pub bit: T,
    pub power: T,
    pub reconstructed_before: T,
    pub reconstructed_after: T,
    pub value: [T; D_EF],
    pub index_lookup_count: T,
}

#[derive(ColumnsAir)]
#[columns_via(NativeShiftScheduleCols<u8>)]
pub struct NativeShiftScheduleAir {
    pub transcript_bus: TranscriptBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub right_shift_bus: RightShiftBus,
    pub shift_index_bus: crate::native_warp::bus::NativeShiftIndexBus,
    pub opening_bus: Option<NativeOpeningClaimBus>,
    pub log_codeword_len: usize,
    pub opening_claim_offset: usize,
}

pub fn generate_native_shift_schedule_trace(
    proof_idx: usize,
    samples: &[F],
    sample_tidx: &[usize],
    indices: &[u32],
    index_lookup_count: u32,
    log_codeword_len: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if samples.is_empty()
        || samples.len() != sample_tidx.len()
        || samples.len() != indices.len()
        || log_codeword_len == 0
        || log_codeword_len > 27
    {
        return None;
    }
    let valid_rows = samples.len() * log_codeword_len;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeShiftScheduleCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mask = (1u32 << log_codeword_len) - 1;
    for shift in 0..samples.len() {
        let canonical = samples[shift].as_canonical_u32();
        if samples[shift] == -F::ONE {
            return None;
        }
        if canonical & mask != indices[shift] {
            return None;
        }
        let quotient = canonical >> log_codeword_len;
        let mut reconstructed = 0u32;
        for coordinate in 0..log_codeword_len {
            let bit_index = log_codeword_len - 1 - coordinate;
            let bit = (indices[shift] >> bit_index) & 1;
            let before = reconstructed;
            reconstructed += bit << bit_index;
            let row_index = shift * log_codeword_len + coordinate;
            let cols: &mut NativeShiftScheduleCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.shift = F::from_usize(shift);
            cols.coordinate = F::from_usize(coordinate);
            cols.is_first = F::from_bool(coordinate == 0);
            cols.is_last = F::from_bool(coordinate + 1 == log_codeword_len);
            cols.is_first_shift = F::from_bool(shift == 0 && coordinate == 0);
            cols.proof_idx = F::from_usize(proof_idx);
            cols.tidx = F::from_usize(sample_tidx[shift]);
            cols.sample = samples[shift];
            cols.accepted_inverse = (samples[shift] + F::ONE).inverse();
            cols.quotient = F::from_u32(quotient);
            cols.index = F::from_u32(indices[shift]);
            cols.bit = F::from_u32(bit);
            cols.power = F::from_u32(1u32 << bit_index);
            cols.reconstructed_before = F::from_u32(before);
            cols.reconstructed_after = F::from_u32(reconstructed);
            cols.value[0] = F::from_u32(bit);
            cols.index_lookup_count = if coordinate == 0 {
                F::from_u32(index_lookup_count)
            } else {
                F::ZERO
            };
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

impl BaseAirWithPublicValues<F> for NativeShiftScheduleAir {}
impl PartitionedBaseAir<F> for NativeShiftScheduleAir {}
impl<F> BaseAir<F> for NativeShiftScheduleAir {
    fn width(&self) -> usize {
        NativeShiftScheduleCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeShiftScheduleAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("native shift schedule row"),
            main.row_slice(1).expect("native shift schedule next row"),
        );
        let local: &NativeShiftScheduleCols<AB::Var> = (*local).borrow();
        let next: &NativeShiftScheduleCols<AB::Var> = (*next).borrow();
        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.is_first_shift,
            local.bit,
        ] {
            builder.assert_bool(flag);
        }
        builder
            .when(local.active * local.is_first_shift)
            .assert_one(local.is_first);
        builder
            .when(local.active * local.is_first_shift)
            .assert_zero(local.shift);
        builder.when_first_row().assert_one(local.active);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.coordinate);
        builder.when(local.active * local.is_last).assert_eq(
            local.coordinate,
            AB::Expr::from_usize(self.log_codeword_len - 1),
        );
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.reconstructed_before);
        builder.when(local.active * local.is_first).assert_eq(
            local.power,
            AB::Expr::from_u32(1u32 << (self.log_codeword_len - 1)),
        );
        builder.when(local.active).assert_eq(
            local.reconstructed_after,
            local.reconstructed_before + local.bit * local.power,
        );
        let same_shift = next.active * (AB::Expr::ONE - next.is_first);
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_shift);
        same.assert_eq(next.proof_idx, local.proof_idx);
        same.assert_zero(next.is_first_shift);
        same.assert_eq(next.shift, local.shift);
        same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        same.assert_eq(next.sample, local.sample);
        same.assert_eq(next.quotient, local.quotient);
        same.assert_eq(next.index, local.index);
        same.assert_eq(local.power, next.power * AB::F::TWO);
        same.assert_eq(next.reconstructed_before, local.reconstructed_after);
        let mut transition = builder.when_transition();
        let mut next_shift = transition.when(next.active * next.is_first);
        next_shift.assert_one(local.is_last);
        next_shift
            .when(next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        next_shift
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx);
        next_shift
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.shift, local.shift + AB::F::ONE);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.reconstructed_after, local.index);
        self.transcript_bus.sample(
            builder,
            local.proof_idx,
            local.tidx,
            local.sample,
            local.active * local.is_first,
        );
        // BabyBear has order 15 * 2^27 + 1. For every supported RS domain
        // (bits <= 27), exact-uniform `sample_bits` rejects precisely the
        // canonical field value p - 1 = -1. Rejected squeezes are certified
        // by `NativeRejectedShiftSampleAir`; this row is the accepted one.
        builder
            .when(local.active * local.is_first)
            .assert_one((local.sample + AB::F::ONE) * local.accepted_inverse);
        self.right_shift_bus.lookup_key(
            builder,
            RightShiftMessage {
                input: local.sample.into(),
                shift_bits: AB::Expr::from_usize(self.log_codeword_len),
                result: local.quotient.into(),
            },
            local.active * local.is_first,
        );
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: AB::Expr::ONE,
                bit_src: local.sample.into(),
                num_bits: AB::Expr::ZERO,
                result: AB::Expr::ONE,
            },
            local.active * local.is_first,
        );
        builder.when(local.active * local.is_first).assert_eq(
            local.sample,
            local.index + local.quotient * AB::Expr::from_u32(1u32 << self.log_codeword_len),
        );
        self.shift_index_bus.add_key_with_lookups(
            builder,
            crate::native_warp::bus::NativeShiftIndexMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                index: local.index.into(),
            },
            local.index_lookup_count,
        );
        builder
            .when(local.active)
            .assert_eq(local.value[0], local.bit);
        for limb in &local.value[1..] {
            builder.when(local.active).assert_zero(*limb);
        }
        if let Some(opening_bus) = self.opening_bus {
            opening_bus.send(
                builder,
                NativeOpeningClaimMessage {
                    proof_idx: local.proof_idx.into(),
                    claim: AB::Expr::from_usize(self.opening_claim_offset) + local.shift,
                    section: AB::Expr::from_usize(OPENING_SECTION_POINT),
                    coordinate: local.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                local.active,
            );
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeRejectedShiftSampleCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub sample: T,
}

/// Transcript rows discarded by exact-uniform `sample_bits` before its final
/// accepted value. For BabyBear and a power-of-two domain of at most 2^27,
/// every discarded value is exactly p - 1.
#[derive(ColumnsAir)]
#[columns_via(NativeRejectedShiftSampleCols<u8>)]
pub struct NativeRejectedShiftSampleAir {
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for NativeRejectedShiftSampleAir {}
impl PartitionedBaseAir<F> for NativeRejectedShiftSampleAir {}
impl BaseAir<F> for NativeRejectedShiftSampleAir {
    fn width(&self) -> usize {
        NativeRejectedShiftSampleCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeRejectedShiftSampleAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native rejected shift sample row");
        let local: &NativeRejectedShiftSampleCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder
            .when(local.active)
            .assert_zero(local.sample + AB::F::ONE);
        self.transcript_bus.sample(
            builder,
            local.proof_idx,
            local.tidx,
            local.sample,
            local.active,
        );
    }
}

#[must_use]
pub fn generate_native_rejected_shift_sample_trace(
    proof_idx: usize,
    samples: &[(usize, F)],
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let valid_rows = samples.len();
    let height = required_height.unwrap_or_else(|| valid_rows.max(1).next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeRejectedShiftSampleCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (row_index, &(tidx, sample)) in samples.iter().enumerate() {
        if sample != -F::ONE {
            return None;
        }
        let cols: &mut NativeRejectedShiftSampleCols<F> =
            trace[row_index * width..(row_index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.tidx = F::from_usize(tidx);
        cols.sample = sample;
    }
    Some(RowMajorMatrix::new(trace, width))
}

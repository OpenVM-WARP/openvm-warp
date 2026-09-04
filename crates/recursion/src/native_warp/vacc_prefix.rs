use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, native_warp::NativeWarpFamilyParams, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::bus::{
        NativeFreshCountBus, NativeFreshCountMessage, NativePcdStateReadBus,
        NativePcdStateReadMessage,
    },
    primitives::bus::{RangeCheckerBus, RangeCheckerBusMessage},
};

const TRANSITION_TAG: &[u8] = b"openvm-native-warp-vacc-transition-v1";
const VACC_TAG: &[u8] = b"openvm-native-warp-vacc-all-ef-v3";
const LIMB_BASE: u32 = 1 << 16;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeVaccPrefixCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub fresh_count: T,
    pub prior_present: T,
    pub is_step_zero: T,
    pub step_zero_inverse: T,
    pub step_lo: T,
    pub step_hi: T,
    pub prior_count_lo: T,
    pub prior_count_hi: T,
    pub multiplication_carry: T,
    pub step_bytes: [T; 8],
    pub fresh_bytes: [T; 8],
    pub prior_bytes: [T; 8],
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub prior_transcript_digest: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(NativeVaccPrefixCols<u8>)]
pub struct NativeVaccPrefixAir {
    pub transcript_bus: TranscriptBus,
    pub state_read_bus: NativePcdStateReadBus,
    pub fresh_count_bus: NativeFreshCountBus,
    pub range_bus: RangeCheckerBus,
    pub family: NativeWarpFamilyParams,
    pub protocol_version: u32,
    pub pvs_app_vk_offset: usize,
    pub pvs_relation_offset: usize,
    pub pvs_segment_count_offset: usize,
    pub pvs_accumulator_is_set_offset: usize,
    pub pvs_transcript_offset: usize,
}

impl BaseAirWithPublicValues<F> for NativeVaccPrefixAir {}
impl PartitionedBaseAir<F> for NativeVaccPrefixAir {}
impl BaseAir<F> for NativeVaccPrefixAir {
    fn width(&self) -> usize {
        NativeVaccPrefixCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeVaccPrefixAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native VACC prefix row");
        let local: &NativeVaccPrefixCols<AB::Var> = (*row).borrow();
        for flag in [local.active, local.prior_present, local.is_step_zero] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder
            .when(local.active)
            .assert_eq(local.proof_idx, local.fresh_count);
        builder
            .when(local.active)
            .assert_eq(local.prior_present, AB::Expr::ONE - local.is_step_zero);
        let step_sum = local.step_lo + local.step_hi;
        builder
            .when(local.active * local.is_step_zero)
            .assert_zero(step_sum.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.is_step_zero))
            .assert_one(step_sum * local.step_zero_inverse);
        // Native OpenVM transitions consume a fixed bounded number of fresh
        // segments. The remaining WARP rows are the prior accumulator and
        // canonical zero padding, so the prior segment count is independent
        // of the generic linear-chain capacity.
        builder.when(local.active).assert_eq(
            local.step_lo * AB::Expr::from_usize(self.family.max_fresh_per_step),
            local.prior_count_lo + local.multiplication_carry * AB::Expr::from_u32(LIMB_BASE),
        );
        builder.when(local.active).assert_eq(
            local.step_hi * AB::Expr::from_usize(self.family.max_fresh_per_step)
                + local.multiplication_carry,
            local.prior_count_hi,
        );
        // The family fixes max_fresh_per_step to one (arity two) or two
        // (arity four). This check permits exactly one or the configured
        // maximum; continuation semantics reject nonterminal partial batches.
        builder.when(local.active).assert_zero(
            (local.fresh_count - AB::F::ONE)
                * (local.fresh_count - AB::F::from_usize(self.family.max_fresh_per_step)),
        );
        for value in [
            local.step_lo,
            local.step_hi,
            local.prior_count_lo,
            local.prior_count_hi,
        ] {
            self.range_bus.lookup_key(
                builder,
                RangeCheckerBusMessage {
                    value: value.into(),
                    max_bits: AB::Expr::from_u8(16),
                },
                local.active,
            );
        }
        self.range_bus.lookup_key(
            builder,
            RangeCheckerBusMessage {
                value: local.multiplication_carry.into(),
                max_bits: AB::Expr::from_u8(8),
            },
            local.active,
        );
        for byte in local
            .step_bytes
            .into_iter()
            .chain(local.fresh_bytes)
            .chain(local.prior_bytes)
        {
            self.range_bus.lookup_key(
                builder,
                RangeCheckerBusMessage {
                    value: byte.into(),
                    max_bits: AB::Expr::from_u8(8),
                },
                local.active,
            );
        }
        builder.when(local.active).assert_eq(
            local.step_lo,
            local.step_bytes[0] + local.step_bytes[1] * AB::Expr::from_u32(256),
        );
        builder.when(local.active).assert_eq(
            local.step_hi,
            local.step_bytes[2] + local.step_bytes[3] * AB::Expr::from_u32(256),
        );
        for byte in &local.step_bytes[4..] {
            builder.when(local.active).assert_zero(*byte);
        }
        builder
            .when(local.active)
            .assert_eq(local.fresh_bytes[0], local.fresh_count);
        builder
            .when(local.active)
            .assert_eq(local.prior_bytes[0], local.prior_present);
        for byte in local.fresh_bytes[1..].iter().chain(&local.prior_bytes[1..]) {
            builder.when(local.active).assert_zero(*byte);
        }
        self.fresh_count_bus.lookup_key(
            builder,
            NativeFreshCountMessage {
                count: local.fresh_count.into(),
            },
            local.active,
        );
        for (coordinate, value) in [
            (self.pvs_segment_count_offset, local.prior_count_lo),
            (self.pvs_segment_count_offset + 1, local.prior_count_hi),
            (self.pvs_accumulator_is_set_offset, local.prior_present),
        ] {
            self.state_read_bus.lookup_key(
                builder,
                NativePcdStateReadMessage {
                    state: AB::Expr::ZERO,
                    coordinate: AB::Expr::from_usize(coordinate),
                    value: value.into(),
                },
                local.active,
            );
        }
        for limb in 0..DIGEST_SIZE {
            for (offset, value) in [
                (self.pvs_app_vk_offset, local.app_vk_digest[limb]),
                (self.pvs_relation_offset, local.relation_digest[limb]),
                (
                    self.pvs_transcript_offset,
                    local.prior_transcript_digest[limb],
                ),
            ] {
                self.state_read_bus.lookup_key(
                    builder,
                    NativePcdStateReadMessage {
                        state: AB::Expr::ZERO,
                        coordinate: AB::Expr::from_usize(offset + limb),
                        value: value.into(),
                    },
                    local.active,
                );
            }
        }
        self.bind_transition_prefix(builder, local);
        self.bind_protocol_prefix(builder, local);
    }
}

impl NativeVaccPrefixAir {
    fn bind_transition_prefix<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &NativeVaccPrefixCols<AB::Var>,
    ) {
        let enabled = AB::Expr::from(local.active);
        let mut tidx = 0usize;
        for value in [F::from_usize(TRANSITION_TAG.len()), F::ZERO] {
            observe_base_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                &mut tidx,
                value,
                enabled.clone(),
            );
        }
        for &byte in TRANSITION_TAG {
            observe_base_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                &mut tidx,
                F::from_u8(byte),
                enabled.clone(),
            );
        }
        for value in [F::from_u32(self.protocol_version), F::ZERO] {
            observe_base_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                &mut tidx,
                value,
                enabled.clone(),
            );
        }
        for digest in [
            local.app_vk_digest,
            local.relation_digest,
            local.prior_transcript_digest,
        ] {
            self.transcript_bus.observe_commit(
                builder,
                local.proof_idx,
                AB::Expr::from_usize(tidx),
                digest,
                enabled.clone(),
            );
            tidx += DIGEST_SIZE;
        }
        self.transcript_bus.observe(
            builder,
            local.proof_idx,
            AB::Expr::from_usize(tidx),
            local.step_lo + local.step_hi * AB::Expr::from_u32(LIMB_BASE),
            enabled.clone(),
        );
        tidx += 1;
        observe_base_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            F::ZERO,
            enabled,
        );
    }

    fn bind_protocol_prefix<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &NativeVaccPrefixCols<AB::Var>,
    ) {
        let enabled = AB::Expr::from(local.active);
        let mut tidx = 2 + TRANSITION_TAG.len() + 2 + 3 * DIGEST_SIZE + 2;
        for &byte in VACC_TAG {
            observe_ext_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                &mut tidx,
                F::from_u8(byte),
                enabled.clone(),
            );
        }
        for bytes in [
            [
                F::ONE,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
            ],
            [F::ZERO; 8],
        ] {
            for byte in bytes {
                observe_ext_const(
                    &self.transcript_bus,
                    builder,
                    local.proof_idx,
                    &mut tidx,
                    byte,
                    enabled.clone(),
                );
            }
        }
        let fixed = [
            self.family.input_arity,
            self.family.num_ood,
            self.family.num_shift_queries,
            self.family.log_message_height,
            self.family.log_codeword_len,
        ];
        for value in fixed {
            for byte in (value as u64).to_le_bytes() {
                observe_ext_const(
                    &self.transcript_bus,
                    builder,
                    local.proof_idx,
                    &mut tidx,
                    F::from_u8(byte),
                    enabled.clone(),
                );
            }
        }
        for bytes in [local.step_bytes, local.fresh_bytes, local.prior_bytes] {
            for byte in bytes {
                self.transcript_bus.observe_ext(
                    builder,
                    local.proof_idx,
                    AB::Expr::from_usize(tidx),
                    [byte.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                    enabled.clone(),
                );
                tidx += D_EF;
            }
        }
    }
}

fn observe_base_const<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: &mut usize,
    value: F,
    enabled: AB::Expr,
) {
    bus.observe(
        builder,
        proof_idx,
        AB::Expr::from_usize(*tidx),
        AB::Expr::from(value),
        enabled,
    );
    *tidx += 1;
}

fn observe_ext_const<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: &mut usize,
    value: F,
    enabled: AB::Expr,
) {
    bus.observe_ext(
        builder,
        proof_idx,
        AB::Expr::from_usize(*tidx),
        [
            AB::Expr::from(value),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ],
        enabled,
    );
    *tidx += D_EF;
}

#[allow(clippy::too_many_arguments)]
pub fn generate_native_vacc_prefix_trace(
    proof_idx: usize,
    fresh_count: usize,
    input_arity: usize,
    max_fresh_per_step: usize,
    prior_count: u32,
    prior_present: bool,
    app_vk_digest: [F; DIGEST_SIZE],
    relation_digest: [F; DIGEST_SIZE],
    prior_transcript_digest: [F; DIGEST_SIZE],
) -> Option<RowMajorMatrix<F>> {
    if input_arity < 2 {
        return None;
    }
    let step = if prior_count == 0 {
        0
    } else {
        if !(prior_count as usize).is_multiple_of(max_fresh_per_step) {
            return None;
        }
        prior_count as usize / max_fresh_per_step
    };
    if prior_present != (step != 0)
        || proof_idx != fresh_count
        || !matches!(max_fresh_per_step, 1 | 2)
        || !(1..=max_fresh_per_step).contains(&fresh_count)
        || max_fresh_per_step + usize::from(prior_present) > input_arity
    {
        return None;
    }
    let width = NativeVaccPrefixCols::<F>::width();
    let mut trace = vec![F::ZERO; width];
    let cols: &mut NativeVaccPrefixCols<F> = trace.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.fresh_count = F::from_usize(fresh_count);
    cols.prior_present = F::from_bool(prior_present);
    cols.is_step_zero = F::from_bool(step == 0);
    let step_lo = step as u32 & 0xffff;
    let step_hi = step as u32 >> 16;
    cols.step_lo = F::from_u32(step_lo);
    cols.step_hi = F::from_u32(step_hi);
    cols.step_zero_inverse = if step == 0 {
        F::ZERO
    } else {
        (cols.step_lo + cols.step_hi).inverse()
    };
    cols.prior_count_lo = F::from_u32(prior_count & 0xffff);
    cols.prior_count_hi = F::from_u32(prior_count >> 16);
    cols.multiplication_carry = F::from_u32((step_lo * max_fresh_per_step as u32) >> 16);
    for (target, byte) in cols.step_bytes.iter_mut().zip((step as u64).to_le_bytes()) {
        *target = F::from_u8(byte);
    }
    cols.fresh_bytes[0] = F::from_usize(fresh_count);
    cols.prior_bytes[0] = F::from_bool(prior_present);
    cols.app_vk_digest = app_vk_digest;
    cols.relation_digest = relation_digest;
    cols.prior_transcript_digest = prior_transcript_digest;
    Some(RowMajorMatrix::new(trace, width))
}

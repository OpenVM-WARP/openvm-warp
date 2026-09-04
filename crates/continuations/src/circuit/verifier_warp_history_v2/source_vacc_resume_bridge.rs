//! Authenticated source-to-VACC transcript resumption.
//!
//! The complete SWIRL verifier and the following WARP/VACC transition share
//! one Fiat--Shamir transcript. Replaying the complete source prefix in every
//! VACC verifier is sound but needlessly quadratic. This AIR consumes the
//! source checkpoint already authenticated by
//! [`SetupPcsSourceProvenanceAirV3`] and routes it to the setup-fixed, private
//! transcript buses of the one shape-batched VACC module.
//!
//! Both messages are required. `ResumeTranscriptStateMessage` initializes the
//! suffix transcript AIR, while `CertifiedTranscriptCheckpointMessage` binds
//! the VACC replay producer's start checkpoint, including the sample cursor.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_recursion_circuit::{
    bus::{
        CertifiedTranscriptCheckpointBus, CertifiedTranscriptCheckpointMessage,
        ResumeTranscriptStateBus, ResumeTranscriptStateMessage,
    },
    native_warp::{
        NativeFixedHLeafVaccRouteBusV4, NativeFixedHLeafVaccRouteMessageV4,
        NativeFixedHLeafVaccRouteV4, FIXED_HLEAF_CONTINUATION_WARP_STEP_V4,
        FIXED_HLEAF_VACC_CAPACITY_V4,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;

use crate::circuit::native_warp_history_v19::{
    SetupPcsSourceProvenanceBusV3, SetupPcsSourceProvenanceMessageV3,
    SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3,
};

/// Setup-fixed destination for one batched VACC package. The proof index is
/// the constrained local slot in the capacity-four HLeaf.
#[derive(Clone, Copy, Debug)]
pub struct VerifierWarpSourceResumeRouteV3 {
    pub resume_state_bus: ResumeTranscriptStateBus,
    pub checkpoint_bus: CertifiedTranscriptCheckpointBus,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierWarpSourceResumeBridgeColsV3<T> {
    pub active: T,
    /// Counts active rows including this row. This setup-fixed countdown
    /// proves that exactly `routes.len()` provenance messages are consumed.
    pub remaining: T,
    pub remaining_inverse: T,
    pub provenance: SetupPcsSourceProvenanceMessageV3<T>,
}

#[derive(Clone, Debug)]
pub struct VerifierWarpSourceResumeBridgeAirV3 {
    pub segment_start: u32,
    pub provenance_bus: SetupPcsSourceProvenanceBusV3,
    pub routes: Arc<[VerifierWarpSourceResumeRouteV3]>,
}

impl BaseAir<F> for VerifierWarpSourceResumeBridgeAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<VerifierWarpSourceResumeBridgeColsV3<u8>>() + self.routes.len()
    }
}
impl BaseAirWithPublicValues<F> for VerifierWarpSourceResumeBridgeAirV3 {}
impl PartitionedBaseAir<F> for VerifierWarpSourceResumeBridgeAirV3 {}

impl<AB> Air<AB> for VerifierWarpSourceResumeBridgeAirV3
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("source resume bridge row");
        let next_row = main.row_slice(1).expect("source resume bridge next row");
        let base_width = core::mem::size_of::<VerifierWarpSourceResumeBridgeColsV3<u8>>();
        let local: &VerifierWarpSourceResumeBridgeColsV3<AB::Var> =
            local_row[..base_width].borrow();
        let next: &VerifierWarpSourceResumeBridgeColsV3<AB::Var> = next_row[..base_width].borrow();
        let route_flags = &local_row[base_width..];

        builder.assert_bool(local.active);
        builder
            .when_first_row()
            .assert_eq(local.remaining, AB::Expr::from_usize(self.routes.len()));
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.remaining);
        builder
            .when_transition()
            .assert_eq(next.remaining, local.remaining - local.active);
        builder.assert_eq(local.remaining * local.remaining_inverse, local.active);
        builder
            .when(AB::Expr::ONE - local.active)
            .assert_zero(local.remaining);

        let mut route_sum = AB::Expr::ZERO;
        let mut route_index = AB::Expr::ZERO;
        for (index, flag) in route_flags.iter().copied().enumerate() {
            builder.assert_bool(flag);
            route_sum += flag.into();
            route_index += AB::Expr::from_usize(index) * flag;
        }
        builder.assert_eq(route_sum, local.active);

        let enabled = local.active;
        builder.when(enabled).assert_eq(
            local.provenance.protocol_version,
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
        );
        let actual_local_transition: AB::Expr = local.provenance.transition_index[0].into();
        let actual_local_transition = actual_local_transition
            + AB::Expr::from_u32(1 << 16)
                * Into::<AB::Expr>::into(local.provenance.transition_index[1]);
        builder
            .when(enabled)
            .assert_eq(actual_local_transition, route_index.clone());
        let expected_segment = AB::Expr::from_u32(self.segment_start) + route_index;
        let actual_segment: AB::Expr = local.provenance.segment_index[0].into();
        let actual_segment = actual_segment
            + AB::Expr::from_u32(1 << 16)
                * Into::<AB::Expr>::into(local.provenance.segment_index[1]);
        builder
            .when(enabled)
            .assert_eq(actual_segment, expected_segment);

        self.provenance_bus
            .receive(builder, local.provenance.clone(), enabled);

        let tidx: AB::Expr = local.provenance.end_tidx[0].into();
        let tidx = tidx
            + AB::Expr::from_u32(1 << 16) * Into::<AB::Expr>::into(local.provenance.end_tidx[1]);
        for (route, flag) in self.routes.iter().zip(route_flags.iter().copied()) {
            route.resume_state_bus.send(
                builder,
                AB::Expr::ZERO,
                ResumeTranscriptStateMessage {
                    tidx: tidx.clone(),
                    state: local.provenance.end_state.map(Into::into),
                },
                flag,
            );
            route.checkpoint_bus.send(
                builder,
                AB::Expr::ZERO,
                CertifiedTranscriptCheckpointMessage {
                    kind: AB::Expr::ZERO,
                    tidx: tidx.clone(),
                    sample_count: local.provenance.end_sample_count.into(),
                    state: local.provenance.end_state.map(Into::into),
                },
                flag,
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierWarpSourceResumeBridgeErrorV3 {
    Empty,
    Count,
    Transition,
    Protocol,
}

pub fn generate_verifier_warp_source_resume_bridge_trace_v3(
    air: &VerifierWarpSourceResumeBridgeAirV3,
    provenances: &[SetupPcsSourceProvenanceMessageV3<F>],
) -> Result<RowMajorMatrix<F>, VerifierWarpSourceResumeBridgeErrorV3> {
    if air.routes.is_empty() {
        return Err(VerifierWarpSourceResumeBridgeErrorV3::Empty);
    }
    if provenances.len() != air.routes.len() {
        return Err(VerifierWarpSourceResumeBridgeErrorV3::Count);
    }
    let base_width = core::mem::size_of::<VerifierWarpSourceResumeBridgeColsV3<u8>>();
    let width = air.width();
    // Keep at least one inactive row so the countdown's terminal zero is
    // constrained even when the route count is itself a power of two.
    let height = (provenances.len() + 1).next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (index, provenance) in provenances.iter().enumerate() {
        if provenance.protocol_version != F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3) {
            return Err(VerifierWarpSourceResumeBridgeErrorV3::Protocol);
        }
        let local =
            u32::try_from(index).map_err(|_| VerifierWarpSourceResumeBridgeErrorV3::Transition)?;
        if provenance.transition_index != [F::from_u32(local & 0xffff), F::from_u32(local >> 16)] {
            return Err(VerifierWarpSourceResumeBridgeErrorV3::Transition);
        }
        let expected_segment = air
            .segment_start
            .checked_add(local)
            .ok_or(VerifierWarpSourceResumeBridgeErrorV3::Transition)?;
        if provenance.segment_index
            != [
                F::from_u32(expected_segment & 0xffff),
                F::from_u32(expected_segment >> 16),
            ]
        {
            return Err(VerifierWarpSourceResumeBridgeErrorV3::Transition);
        }
        let row = &mut values[index * width..(index + 1) * width];
        let cols: &mut VerifierWarpSourceResumeBridgeColsV3<F> = row[..base_width].borrow_mut();
        cols.active = F::ONE;
        cols.remaining = F::from_usize(provenances.len() - index);
        cols.remaining_inverse = cols.remaining.inverse();
        cols.provenance = provenance.clone();
        row[base_width + index] = F::ONE;
    }
    Ok(RowMajorMatrix::new(values, width))
}

/// Setup-fixed local slot table for the capacity-four HLeaf.  Occupancy and
/// the History-global transition index live only in the main trace.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierWarpSourceResumeBridgePrepColsV4<T> {
    pub local_slot: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierWarpSourceResumeBridgeColsV4<T> {
    pub active: T,
    pub node_index: [T; 2],
    pub node_index_bits: [[T; 16]; 2],
    pub global_transition_index: [T; 2],
    pub global_transition_bits: [[T; 16]; 2],
    pub is_global_zero: T,
    /// Inverse of the sum of all 32 constrained global-index bits.  Since the
    /// sum is at most 32, it cannot wrap in BabyBear.
    pub global_bit_sum_inverse: T,
    pub provenance: SetupPcsSourceProvenanceMessageV3<T>,
}

/// Fixed all-continuation dispatcher for one HLeaf.
///
/// Every real transition is verified by one shape-batched, prior-bearing VACC
/// package. At global transition zero the prior is the setup-authenticated
/// valid seed; afterward it is the preceding real output.
#[derive(Clone, Debug)]
pub struct VerifierWarpSourceResumeBridgeAirV4 {
    pub provenance_bus: SetupPcsSourceProvenanceBusV3,
    pub route: VerifierWarpSourceResumeRouteV3,
    pub route_bus: NativeFixedHLeafVaccRouteBusV4,
    /// Fixed number of downstream constraints which consume the canonical
    /// route message (prefix/statement/replay adapters in the final assembly).
    pub route_lookup_count: usize,
}

impl BaseAir<F> for VerifierWarpSourceResumeBridgeAirV4 {
    fn width(&self) -> usize {
        core::mem::size_of::<VerifierWarpSourceResumeBridgeColsV4<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<VerifierWarpSourceResumeBridgePrepColsV4<u8>>();
        let mut values = F::zero_vec(width * FIXED_HLEAF_VACC_CAPACITY_V4);
        for slot in 0..FIXED_HLEAF_VACC_CAPACITY_V4 {
            let cols: &mut VerifierWarpSourceResumeBridgePrepColsV4<F> =
                values[slot * width..(slot + 1) * width].borrow_mut();
            cols.local_slot = F::from_usize(slot);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for VerifierWarpSourceResumeBridgeAirV4 {}
impl PartitionedBaseAir<F> for VerifierWarpSourceResumeBridgeAirV4 {}

impl<AB> Air<AB> for VerifierWarpSourceResumeBridgeAirV4
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.route_lookup_count != 0, "unconsumed HLeaf VACC route");
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("source resume v4 prep row");
        let prep: &VerifierWarpSourceResumeBridgePrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("source resume v4 row");
        let next_row = main.row_slice(1).expect("source resume v4 next row");
        let local: &VerifierWarpSourceResumeBridgeColsV4<AB::Var> = (*row).borrow();
        let next: &VerifierWarpSourceResumeBridgeColsV4<AB::Var> = (*next_row).borrow();

        builder.assert_bool(local.active);
        builder.assert_bool(local.is_global_zero);
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_zero(next.active * (AB::Expr::ONE - local.active));
        for value in row.iter().skip(1) {
            builder
                .when(AB::Expr::ONE - local.active)
                .assert_zero(*value);
        }

        let mut global_bit_sum = AB::Expr::ZERO;
        for (limb, bits) in local
            .global_transition_index
            .into_iter()
            .zip(local.global_transition_bits)
        {
            let mut reconstructed = AB::Expr::ZERO;
            for (bit_index, bit) in bits.into_iter().enumerate() {
                builder.when(local.active).assert_bool(bit);
                reconstructed += bit * AB::Expr::from_u32(1 << bit_index);
                global_bit_sum += bit;
            }
            builder.when(local.active).assert_eq(limb, reconstructed);
        }
        builder
            .when(local.active * local.is_global_zero)
            .assert_zero(global_bit_sum.clone());
        builder
            .when(local.active * local.is_global_zero)
            .assert_zero(local.global_bit_sum_inverse);
        builder
            .when(local.active * (AB::Expr::ONE - local.is_global_zero))
            .assert_one(global_bit_sum * local.global_bit_sum_inverse);

        for (limb, bits) in local.node_index.into_iter().zip(local.node_index_bits) {
            let mut reconstructed = AB::Expr::ZERO;
            for (bit_index, bit) in bits.into_iter().enumerate() {
                builder.when(local.active).assert_bool(bit);
                reconstructed += bit * AB::Expr::from_u32(1 << bit_index);
            }
            builder.when(local.active).assert_eq(limb, reconstructed);
        }

        // Full integer identity, proved bitwise without embedding a u32 into
        // BabyBear: global bits [2..32] are node bits [0..30], and the two
        // top node bits are zero. Together with the low slot bits below this
        // is exactly `global = 4 * node + slot` over u32.
        for bit in 0..30 {
            let node_bit = local.node_index_bits[bit / 16][bit % 16];
            let global_bit = local.global_transition_bits[(bit + 2) / 16][(bit + 2) % 16];
            builder.when(local.active).assert_eq(node_bit, global_bit);
        }
        builder
            .when(local.active)
            .assert_zero(local.node_index_bits[1][14]);
        builder
            .when(local.active)
            .assert_zero(local.node_index_bits[1][15]);

        // `global = 4 * node + local_slot`: for capacity four the low two
        // bits are exactly the setup-fixed slot bits.
        builder.when(local.active).assert_eq(
            local.global_transition_bits[0][0],
            AB::Expr::from(prep.local_slot) - AB::Expr::TWO * local.global_transition_bits[0][1],
        );
        builder.when(local.active).assert_zero(
            prep.local_slot
                * (prep.local_slot - AB::Expr::ONE)
                * (prep.local_slot - AB::Expr::TWO)
                * (prep.local_slot - AB::Expr::from_u32(3)),
        );

        let enabled = AB::Expr::from(local.active);
        builder.when(enabled.clone()).assert_eq(
            local.provenance.protocol_version,
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
        );
        builder
            .when(enabled.clone())
            .assert_eq(local.provenance.transition_index[0], prep.local_slot);
        builder
            .when(enabled.clone())
            .assert_zero(local.provenance.transition_index[1]);
        builder
            .when(enabled.clone())
            .assert_eq(local.provenance.segment_index[0], prep.local_slot);
        builder
            .when(enabled.clone())
            .assert_zero(local.provenance.segment_index[1]);
        self.provenance_bus
            .receive(builder, local.provenance.clone(), enabled.clone());

        let tidx: AB::Expr = local.provenance.end_tidx[0].into();
        let tidx = tidx
            + AB::Expr::from_u32(1 << 16) * Into::<AB::Expr>::into(local.provenance.end_tidx[1]);
        let checkpoint = CertifiedTranscriptCheckpointMessage {
            kind: AB::Expr::ZERO,
            tidx: tidx.clone(),
            sample_count: local.provenance.end_sample_count.into(),
            state: local.provenance.end_state.map(Into::into),
        };
        let resume = ResumeTranscriptStateMessage {
            tidx,
            state: local.provenance.end_state.map(Into::into),
        };
        self.route.resume_state_bus.send(
            builder,
            prep.local_slot.into(),
            resume.clone(),
            enabled.clone(),
        );
        self.route.checkpoint_bus.send(
            builder,
            prep.local_slot.into(),
            checkpoint,
            enabled.clone(),
        );

        self.route_bus.add_key_with_lookups(
            builder,
            NativeFixedHLeafVaccRouteMessageV4 {
                local_slot: prep.local_slot.into(),
                node_index_lo: local.node_index[0].into(),
                node_index_hi: local.node_index[1].into(),
                global_transition_index_lo: local.global_transition_index[0].into(),
                global_transition_index_hi: local.global_transition_index[1].into(),
                module_proof_idx: prep.local_slot.into(),
                has_prior: AB::Expr::ONE,
                warp_step: AB::Expr::from_u32(FIXED_HLEAF_CONTINUATION_WARP_STEP_V4),
            },
            enabled * AB::Expr::from_usize(self.route_lookup_count),
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifierWarpSourceResumeBridgeErrorV4 {
    Empty,
    Capacity,
    GlobalStart,
    Transition(usize),
    Protocol(usize),
}

#[derive(Debug)]
pub struct VerifierWarpSourceResumeBridgeTraceV4 {
    pub matrix: RowMajorMatrix<F>,
    pub routes: Vec<NativeFixedHLeafVaccRouteV4>,
}

pub fn generate_verifier_warp_source_resume_bridge_trace_v4(
    global_transition_start: u32,
    provenances: &[SetupPcsSourceProvenanceMessageV3<F>],
) -> Result<VerifierWarpSourceResumeBridgeTraceV4, VerifierWarpSourceResumeBridgeErrorV4> {
    if provenances.is_empty() {
        return Err(VerifierWarpSourceResumeBridgeErrorV4::Empty);
    }
    if provenances.len() > FIXED_HLEAF_VACC_CAPACITY_V4 {
        return Err(VerifierWarpSourceResumeBridgeErrorV4::Capacity);
    }
    if global_transition_start % FIXED_HLEAF_VACC_CAPACITY_V4 as u32 != 0 {
        return Err(VerifierWarpSourceResumeBridgeErrorV4::GlobalStart);
    }
    let width = core::mem::size_of::<VerifierWarpSourceResumeBridgeColsV4<u8>>();
    let mut values = F::zero_vec(width * FIXED_HLEAF_VACC_CAPACITY_V4);
    let mut routes = Vec::with_capacity(provenances.len());
    for (slot, provenance) in provenances.iter().enumerate() {
        if provenance.protocol_version != F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3) {
            return Err(VerifierWarpSourceResumeBridgeErrorV4::Protocol(slot));
        }
        let local = [F::from_usize(slot), F::ZERO];
        if provenance.transition_index != local || provenance.segment_index != local {
            return Err(VerifierWarpSourceResumeBridgeErrorV4::Transition(slot));
        }
        let global = global_transition_start
            .checked_add(slot as u32)
            .ok_or(VerifierWarpSourceResumeBridgeErrorV4::GlobalStart)?;
        let route = NativeFixedHLeafVaccRouteV4::derive(slot, global, slot as u32)
            .map_err(|_| VerifierWarpSourceResumeBridgeErrorV4::Transition(slot))?;
        let cols: &mut VerifierWarpSourceResumeBridgeColsV4<F> =
            values[slot * width..(slot + 1) * width].borrow_mut();
        cols.active = F::ONE;
        let node = global / FIXED_HLEAF_VACC_CAPACITY_V4 as u32;
        cols.node_index = [F::from_u32(node & 0xffff), F::from_u32(node >> 16)];
        cols.global_transition_index = [F::from_u32(global & 0xffff), F::from_u32(global >> 16)];
        for (limb_index, limb) in [global as u16, (global >> 16) as u16]
            .into_iter()
            .enumerate()
        {
            for bit in 0..16 {
                cols.global_transition_bits[limb_index][bit] =
                    F::from_bool(((limb >> bit) & 1) == 1);
            }
        }
        for (limb_index, limb) in [node as u16, (node >> 16) as u16].into_iter().enumerate() {
            for bit in 0..16 {
                cols.node_index_bits[limb_index][bit] = F::from_bool(((limb >> bit) & 1) == 1);
            }
        }
        cols.is_global_zero = F::from_bool(global == 0);
        let bit_sum = global.count_ones();
        cols.global_bit_sum_inverse = if bit_sum == 0 {
            F::ZERO
        } else {
            F::from_u32(bit_sum).inverse()
        };
        cols.provenance = provenance.clone();
        routes.push(route);
    }
    Ok(VerifierWarpSourceResumeBridgeTraceV4 {
        matrix: RowMajorMatrix::new(values, width),
        routes,
    })
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_recursion_circuit::{
        native_warp::{NativeFixedHLeafVaccRouteBusV4, FIXED_HLEAF_VACC_CAPACITY_V4},
        system::BusIndexManager,
    };
    use openvm_stark_backend::{
        air_builders::debug::check_constraints,
        p3_field::{PrimeCharacteristicRing, PrimeField32},
        p3_matrix::Matrix,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        BabyBearPoseidon2Config as NativeSC, DIGEST_SIZE,
    };

    use super::*;
    use crate::circuit::native_warp_history_v19::TRANSCRIPT_WIDTH_V19;

    fn provenance(slot: usize) -> SetupPcsSourceProvenanceMessageV3<F> {
        SetupPcsSourceProvenanceMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_index: [F::from_usize(slot), F::ZERO],
            segment_index: [F::from_usize(slot), F::ZERO],
            app_vk_digest: [F::ZERO; DIGEST_SIZE],
            relation_digest: [F::ZERO; DIGEST_SIZE],
            source_root: [F::ZERO; DIGEST_SIZE],
            source_instance_digest: [F::ZERO; DIGEST_SIZE],
            source_forest_root: [F::ZERO; DIGEST_SIZE],
            segment_openings_digest: [F::ZERO; DIGEST_SIZE],
            source_checkpoint_digest: [F::ZERO; DIGEST_SIZE],
            source_manifest_digest: [F::ZERO; DIGEST_SIZE],
            source_receipt_digest: [F::ZERO; DIGEST_SIZE],
            end_tidx: [F::from_usize(10 + slot), F::ZERO],
            end_sample_count: F::ZERO,
            end_state: [F::ZERO; TRANSCRIPT_WIDTH_V19],
        }
    }

    fn air() -> VerifierWarpSourceResumeBridgeAirV4 {
        let mut manager = BusIndexManager::new();
        let route = VerifierWarpSourceResumeRouteV3 {
            resume_state_bus: ResumeTranscriptStateBus::new(manager.new_bus_idx()),
            checkpoint_bus: CertifiedTranscriptCheckpointBus::new(manager.new_bus_idx()),
        };
        VerifierWarpSourceResumeBridgeAirV4 {
            provenance_bus: SetupPcsSourceProvenanceBusV3::new(manager.new_bus_idx()),
            route,
            route_bus: NativeFixedHLeafVaccRouteBusV4::new(manager.new_bus_idx()),
            route_lookup_count: 1,
        }
    }

    fn check(air: &VerifierWarpSourceResumeBridgeAirV4, matrix: &RowMajorMatrix<F>) {
        let prep = air.preprocessed_trace().expect("fixed prep");
        check_constraints::<_, NativeSC>(
            air,
            "VerifierWarpSourceResumeBridgeAirV4",
            &Some(prep.as_view()),
            &[matrix.as_view()],
            &[],
        );
    }

    #[test]
    fn genesis_and_later_slot_zero_are_seeded_continuations() {
        let air = air();
        let provenances = (0..FIXED_HLEAF_VACC_CAPACITY_V4)
            .map(provenance)
            .collect::<Vec<_>>();
        let genesis = generate_verifier_warp_source_resume_bridge_trace_v4(0, &provenances)
            .expect("genesis routes");
        check(&air, &genesis.matrix);
        assert!(genesis.routes.iter().all(|route| route.has_prior));
        assert!(genesis.routes.iter().all(|route| route.warp_step == 1));
        assert!(genesis
            .routes
            .iter()
            .enumerate()
            .all(|(slot, route)| route.module_proof_idx == slot as u32));

        let later = generate_verifier_warp_source_resume_bridge_trace_v4(8, &provenances)
            .expect("later routes");
        check(&air, &later.matrix);
        assert_eq!(later.routes[0].local_slot, 0);
        assert_eq!(later.routes[0].node_index, 2);
        assert!(later.routes[0].has_prior);
        assert_eq!(later.routes[0].warp_step, 1);
    }

    #[test]
    fn field_modulus_cannot_alias_global_zero() {
        // BabyBear p == 1 (mod 4), so slot one can carry the exact integer p.
        let start = F::ORDER_U32 - 1;
        let provenances = (0..2).map(provenance).collect::<Vec<_>>();
        let mut trace = generate_verifier_warp_source_resume_bridge_trace_v4(start, &provenances)
            .expect("modulus crossing trace");
        check(&air(), &trace.matrix);
        let width = trace.matrix.width();
        let row: &mut VerifierWarpSourceResumeBridgeColsV4<F> =
            trace.matrix.values[width..2 * width].borrow_mut();
        assert_eq!(
            row.global_transition_index,
            [
                F::from_u32(F::ORDER_U32 & 0xffff),
                F::from_u32(F::ORDER_U32 >> 16)
            ]
        );
        row.is_global_zero = F::ONE;
        row.global_bit_sum_inverse = F::ZERO;
        assert!(catch_unwind(AssertUnwindSafe(|| check(&air(), &trace.matrix))).is_err());
    }

    #[test]
    fn selector_and_full_node_relation_mutations_fail() {
        let provenances = (0..2).map(provenance).collect::<Vec<_>>();
        let original = generate_verifier_warp_source_resume_bridge_trace_v4(4, &provenances)
            .expect("later trace");
        let width = original.matrix.width();

        let mut selector = original.matrix.clone();
        let first: &mut VerifierWarpSourceResumeBridgeColsV4<F> =
            selector.values[..width].borrow_mut();
        first.is_global_zero = F::ONE;
        first.global_bit_sum_inverse = F::ZERO;
        assert!(catch_unwind(AssertUnwindSafe(|| check(&air(), &selector))).is_err());

        let mut node = original.matrix.clone();
        let first: &mut VerifierWarpSourceResumeBridgeColsV4<F> = node.values[..width].borrow_mut();
        first.node_index[0] += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| check(&air(), &node))).is_err());

        assert!(generate_verifier_warp_source_resume_bridge_trace_v4(1, &provenances).is_err());
    }

    #[test]
    fn inactive_suffix_and_key_shape_are_occupancy_invariant() {
        let air = air();
        let prep = air.preprocessed_trace().expect("prep");
        let expected_width = air.width();
        for occupancy in 1..=FIXED_HLEAF_VACC_CAPACITY_V4 {
            let provenances = (0..occupancy).map(provenance).collect::<Vec<_>>();
            let trace = generate_verifier_warp_source_resume_bridge_trace_v4(0, &provenances)
                .expect("fixed trace");
            assert_eq!(trace.matrix.width(), expected_width);
            assert_eq!(trace.matrix.height(), FIXED_HLEAF_VACC_CAPACITY_V4);
            assert!(trace.matrix.values[occupancy * expected_width..]
                .iter()
                .all(|value| *value == F::ZERO));
            assert_eq!(
                air.preprocessed_trace().expect("prep repeat").values,
                prep.values
            );
            check(&air, &trace.matrix);
        }
    }
}

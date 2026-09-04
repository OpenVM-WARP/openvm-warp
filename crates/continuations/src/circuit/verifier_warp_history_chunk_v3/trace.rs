//! Deterministic witness generation for the two-child composition AIR.

use core::borrow::BorrowMut;

use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::PrimeCharacteristicRing, p3_matrix::dense::RowMajorMatrix,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;

use super::{
    VerifierWarpHistoryChunkCompositionAirV3, VerifierWarpHistoryChunkCompositionColsV3,
    VerifierWarpHistoryChunkCompositionErrorV3, VerifierWarpHistoryChunkCompositionOutputV3,
    VerifierWarpHistoryChunkCompositionRecordV3, VerifierWarpHistoryChunkIntervalMessageV3,
};
use crate::circuit::verifier_warp_history_v2::VerifierWarpHistoryChunkPublicValuesV3;

pub struct VerifierWarpHistoryChunkCompositionTraceV3 {
    pub matrix: RowMajorMatrix<F>,
    pub public_values: Vec<F>,
    pub merged: VerifierWarpHistoryChunkPublicValuesV3,
}

impl VerifierWarpHistoryChunkCompositionAirV3 {
    pub fn generate_trace(
        &self,
        record: &VerifierWarpHistoryChunkCompositionRecordV3,
    ) -> Result<
        VerifierWarpHistoryChunkCompositionTraceV3,
        VerifierWarpHistoryChunkCompositionErrorV3,
    > {
        let merged = record.merged()?;
        let width = self.width();
        let mut values = F::zero_vec(2 * width);
        let local: &mut VerifierWarpHistoryChunkCompositionColsV3<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        local.chunk_index_carry = F::from_bool((record.left.chunk_index as u16) == u16::MAX);
        set_u32_bits(&mut local.left_chunk_index_bits, record.left.chunk_index);
        set_u32_bits(&mut local.right_chunk_index_bits, record.right.chunk_index);
        local.left = VerifierWarpHistoryChunkIntervalMessageV3::from_public_values(record.left);
        local.right = VerifierWarpHistoryChunkIntervalMessageV3::from_public_values(record.right);
        let public_values = match self.output_mode() {
            VerifierWarpHistoryChunkCompositionOutputV3::StandalonePublicValues => merged.to_vec(),
            VerifierWarpHistoryChunkCompositionOutputV3::TypedBus { .. } => Vec::new(),
        };
        Ok(VerifierWarpHistoryChunkCompositionTraceV3 {
            matrix: RowMajorMatrix::new(values, width),
            public_values,
            merged,
        })
    }
}

fn set_u32_bits(bits: &mut [[F; 16]; 2], value: u32) {
    for (limb_index, limb_bits) in bits.iter_mut().enumerate() {
        let limb = ((value >> (16 * limb_index)) & 0xffff) as u16;
        for (bit_index, bit) in limb_bits.iter_mut().enumerate() {
            *bit = F::from_bool(((limb >> bit_index) & 1) == 1);
        }
    }
}

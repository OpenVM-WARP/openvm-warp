//! Canonical parameter binding shared by the reduced-SWIRL recursive components.

use openvm_stark_backend::{
    hasher::MerkleHasher, p3_field::PrimeCharacteristicRing, StarkProtocolConfig, SystemParams,
    WhirProximityStrategy,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, F};

use crate::SC;

const REDUCED_SWIRL_SYSTEM_PARAMS_DIGEST_TAG: u64 = 0x5657_5350_5359_0301;

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlParamsDigestError {
    #[error("system parameters do not match the supplied STARK configuration")]
    SystemParameters,
    #[error("system parameter does not fit the canonical wire format")]
    Encoding,
}

/// Bind every parameter that changes the RS, WHIR, or LogUp semantics used by
/// a reduced-SWIRL component. The encoding is stable across serde formats and
/// includes the complete per-round WHIR query schedule.
pub fn reduced_swirl_system_params_digest(
    config: &SC,
    params: &SystemParams,
) -> Result<Digest, ReducedSwirlParamsDigestError> {
    if config.params() != params {
        return Err(ReducedSwirlParamsDigestError::SystemParameters);
    }
    let mut fields = Vec::new();
    push_u64(&mut fields, REDUCED_SWIRL_SYSTEM_PARAMS_DIGEST_TAG);
    push_u64(&mut fields, u64_from_usize(params.l_skip)?);
    push_u64(&mut fields, u64_from_usize(params.n_stack)?);
    push_u64(&mut fields, u64_from_usize(params.w_stack)?);
    push_u64(&mut fields, u64_from_usize(params.log_blowup)?);
    push_u64(
        &mut fields,
        u64_from_usize(params.log_commit_rows_per_query)?,
    );
    push_u64(&mut fields, u64_from_usize(params.max_constraint_degree)?);
    push_u64(&mut fields, u64::from(params.logup.max_interaction_count));
    push_u64(&mut fields, u64::from(params.logup.log_max_message_length));
    push_u64(&mut fields, u64_from_usize(params.logup.pow_bits)?);
    push_u64(&mut fields, u64_from_usize(params.whir.k)?);
    push_u64(&mut fields, u64_from_usize(params.whir.mu_pow_bits)?);
    push_u64(
        &mut fields,
        u64_from_usize(params.whir.query_phase_pow_bits)?,
    );
    push_u64(&mut fields, u64_from_usize(params.whir.folding_pow_bits)?);
    match params.whir.proximity {
        WhirProximityStrategy::UniqueDecoding => push_u64(&mut fields, 0),
        WhirProximityStrategy::SplitUniqueList {
            m,
            list_start_round,
        } => {
            push_u64(&mut fields, 1);
            push_u64(&mut fields, u64_from_usize(m)?);
            push_u64(&mut fields, u64_from_usize(list_start_round)?);
        }
        WhirProximityStrategy::ListDecoding { m } => {
            push_u64(&mut fields, 2);
            push_u64(&mut fields, u64_from_usize(m)?);
        }
    }
    push_u64(&mut fields, u64_from_usize(params.whir.rounds.len())?);
    for round in &params.whir.rounds {
        push_u64(&mut fields, u64_from_usize(round.num_queries)?);
    }
    Ok(config.hasher().hash_slice(&fields))
}

fn u64_from_usize(value: usize) -> Result<u64, ReducedSwirlParamsDigestError> {
    u64::try_from(value).map_err(|_| ReducedSwirlParamsDigestError::Encoding)
}

fn push_u64(fields: &mut Vec<F>, value: u64) {
    fields.push(F::from_u32(value as u32));
    fields.push(F::from_u32((value >> 32) as u32));
}

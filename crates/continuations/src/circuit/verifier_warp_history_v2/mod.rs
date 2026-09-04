//! Succinct History state machine for the fixed capacity-four verifier-WARP
//! relation.
//!
//! Capacity four describes the complete SWIRL segment proofs checked by one
//! fresh source. A streaming WARP transition still has exactly two inputs: that fresh source
//! and the prior accumulator.  The AIR keeps these protocol dimensions
//! separate and receives source and VACC statements only over typed buses.

mod active_count_functional;
mod air;
mod authority_bridge;
mod block_public_values;
mod fixed_capacity_setup_authority_v4;
mod fixed_setup_opening;
mod inventory;
mod ordered_stacking_source_point;
mod ordered_stacking_statement;
mod producer_bridge;
mod record;
mod setup_pcs_authority_bridge_v3;
mod setup_pcs_authority_completion_v3;
mod setup_pcs_authority_statement_v3;
mod setup_pcs_source_provenance_v3;
mod source_functional;
mod source_vacc_resume_bridge;
mod trace;

#[cfg(test)]
mod authority_integration_tests;
#[cfg(test)]
mod chunk_tests;
#[cfg(test)]
mod ordered_stacking_tests;
#[cfg(test)]
mod setup_pcs_authority_statement_tests;
#[cfg(test)]
mod source_provenance_tests;
#[cfg(test)]
mod tests;

pub use active_count_functional::*;
pub use air::*;
pub use authority_bridge::*;
pub use block_public_values::*;
pub use fixed_capacity_setup_authority_v4::*;
pub use fixed_setup_opening::*;
pub use inventory::*;
pub use ordered_stacking_source_point::*;
pub use ordered_stacking_statement::*;
pub use producer_bridge::*;
pub use record::*;
pub use setup_pcs_authority_bridge_v3::*;
pub use setup_pcs_authority_completion_v3::*;
pub use setup_pcs_authority_statement_v3::*;
pub use setup_pcs_source_provenance_v3::*;
pub use source_functional::*;
pub use source_vacc_resume_bridge::*;
pub use trace::*;

pub const VERIFIER_WARP_HISTORY_PROTOCOL_V2: u32 = 2;
/// Version of the recursively composable History interval statement.  The
/// authenticated source and VACC certificates remain protocol-v2 objects;
/// v3 changes only the History boundary from one genesis-to-terminal run to
/// an explicitly chained interval.
pub const VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3: u32 = 3;
/// Number of complete SWIRL segment proofs verified by one fresh WARP source.
///
/// Capacity three keeps the fixed verifier relation below the `2^28`
/// systematic-message boundary on the production Reth profile.  Capacity four
/// crossed that boundary by only 5.47%, but power-of-two padding doubled every
/// resident EF4 accumulator and made an ordinary prior/source/output WARP
/// transition impossible on a 32 GiB device.
pub const VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2: usize = 3;
/// Number of inputs to an ordinary WARP accumulation transition: fresh plus
/// prior accumulator (or canonical padding for the first transition).
pub const VERIFIER_WARP_VACC_INPUT_ARITY_V2: usize = 2;

pub const HISTORY_START_TAG_V2: u32 = 0x5657_4801;
pub const HISTORY_END_TAG_V2: u32 = 0x5657_4802;
pub const MANIFEST_START_TAG_V2: u32 = 0x5657_4803;
pub const MANIFEST_HEADER_TAG_V2: u32 = 0x5657_4804;
pub const MANIFEST_CHILDREN_TAG_V2: u32 = 0x5657_4805;
pub const MANIFEST_CHILD_TAG_V2: u32 = 0x5657_4806;
pub const MANIFEST_ACCUMULATORS_TAG_V2: u32 = 0x5657_4807;
pub const MANIFEST_SOURCE_TAG_V2: u32 = 0x5657_4808;
pub const MANIFEST_END_TAG_V2: u32 = 0x5657_4809;
pub const MANIFEST_PUBLIC_VALUES_TAG_V2: u32 = 0x5657_480a;
pub const MANIFEST_SOURCE_AUTH_TAG_V2: u32 = 0x5657_480b;

pub const STATEMENT_DIGEST_START_TAG_V2: u32 = 0x5657_5301;
pub const STATEMENT_DIGEST_FIELD_TAG_V2: u32 = 0x5657_5302;
pub const STATEMENT_DIGEST_COMMITMENT_TAG_V2: u32 = 0x5657_5303;
pub const STATEMENT_DIGEST_END_TAG_V2: u32 = 0x5657_5304;

const MANIFEST_BASE_OBSERVATION_COUNT_V2: usize = 27;
const MANIFEST_CHILD_OBSERVATION_COUNT_V2: usize = 9;
pub const MANIFEST_OBSERVATION_COUNT_V2: usize = MANIFEST_BASE_OBSERVATION_COUNT_V2
    + VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 * MANIFEST_CHILD_OBSERVATION_COUNT_V2;
const MANIFEST_BASE_HASH_STEP_COUNT_V2: usize = 40;
const MANIFEST_CHILD_HASH_STEP_COUNT_V2: usize = 11;
pub const MANIFEST_HASH_STEP_COUNT_V2: usize = MANIFEST_BASE_HASH_STEP_COUNT_V2
    + VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 * MANIFEST_CHILD_HASH_STEP_COUNT_V2;
pub const HISTORY_HEADER_OBSERVATION_COUNT_V2: usize = 6;
pub const HISTORY_HEADER_HASH_STEP_COUNT_V2: usize = 6;
pub const HISTORY_TRAILER_HASH_STEP_COUNT_V2: usize = 2;
pub const HISTORY_ROW_HASH_STEP_COUNT_V2: usize = HISTORY_HEADER_HASH_STEP_COUNT_V2
    + MANIFEST_HASH_STEP_COUNT_V2
    + HISTORY_TRAILER_HASH_STEP_COUNT_V2;

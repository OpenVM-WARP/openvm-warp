//! Recursive verifier primitives for the genuine complete fixed-multi-AIR
//! terminal relation.
//!
//! This module is intentionally separate from `fixed_multi_air`: complete
//! terminal proofs include local AIR, beta-power, inverse, and global LogUp
//! obligations and cannot soundly be converted to the legacy local-only proof.

mod accumulator_bridge;
mod block_eq;
mod bus;
mod circuit;
mod global_mapping;
mod interaction_endpoint;
mod local_endpoint;
mod prefix_decomposition;
mod region_cursor_openings;
mod sumcheck;
mod trace;

#[cfg(test)]
mod trace_tests;

pub use accumulator_bridge::*;
pub use block_eq::*;
pub use bus::*;
pub use circuit::*;
pub use global_mapping::*;
pub use interaction_endpoint::*;
pub use local_endpoint::*;
pub use prefix_decomposition::*;
pub use region_cursor_openings::*;
pub use sumcheck::*;
pub use trace::*;

//! Two-child recursive composition for authenticated verifier-WARP History intervals.
//!
//! This module deliberately stops at the interval-composition boundary.  A
//! later adapter may connect the two input buses to generic recursive proof
//! verifiers, but this AIR never accepts an opaque History digest: each child
//! message contains all 108 fields of
//! [`VerifierWarpHistoryChunkPublicValuesV3`].

mod air;
mod bus;
mod final_endpoint;
mod record;
mod recursive_bridge;
mod terminal_root;
mod trace;

pub use air::*;
pub use bus::*;
pub use final_endpoint::*;
pub use record::*;
pub use recursive_bridge::*;
pub use terminal_root::*;
pub use trace::*;

#[cfg(test)]
mod tests;

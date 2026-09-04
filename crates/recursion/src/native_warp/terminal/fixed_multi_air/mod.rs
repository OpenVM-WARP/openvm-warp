//! Bounded-width recursive verifier producers for the genuine fixed multi-AIR
//! terminal Decide.
//!
//! These producers replay backend terminal protocol version one.  Relation
//! schedules and symbolic instructions are cached/preprocessed rows; proof
//! messages and recurrence witnesses are common-main rows.  No producer in
//! this module implements the retired `mu + xi * (eta + beta_last)` shortcut.

mod bus;
mod circuit;
mod decomposition;
mod endpoint;
mod endpoint_fixed;
mod endpoint_weight;
mod linearizer_adjoint;
mod mapped;
mod padding;
mod prefix;
mod region_tail;
mod statement;
mod sumcheck;
mod trace;

pub use bus::*;
pub use circuit::*;
pub use decomposition::*;
pub use endpoint::*;
pub use endpoint_fixed::*;
pub use endpoint_weight::*;
pub use linearizer_adjoint::*;
pub use mapped::*;
pub use padding::*;
pub use prefix::*;
pub use region_tail::*;
pub use statement::*;
pub use sumcheck::*;
pub use trace::*;

/// Backend fixed multi-AIR terminal transcript version implemented here.
pub const FIXED_MULTI_AIR_RECURSIVE_TERMINAL_VERSION: u32 = 1;

#[cfg(test)]
mod tests;

//! Fixed-capacity recursive boundary for finite verifier-WARP v3.
//!
//! This module intentionally contains only the reusable wrapper boundary.  It
//! does not claim that a host-verified WARP proof is a recursive proof.  The
//! linkage AIR consumes authenticated receipts from three missing producer
//! components: the ordered-manifest verifier, the ordinary WARP `Verify`
//! calls, and terminal `Decide`.  Until those producer AIRs are supplied, the
//! SDK proving API fails closed.
//!
//! The public AIR order is compatible with OpenVM recursion:
//!
//! - AIR 0 exposes [`VerifierBasePvs`](openvm_verify_stark_host::pvs::VerifierBasePvs);
//! - AIR 1 exposes [`VmPvs`](openvm_verify_stark_host::pvs::VmPvs);
//! - AIR 2 keeps the complete [`FiniteWarpV3PublicStatement`] private and constrains it against AIR
//!   1 through the execution bus and against the authenticated Verify/Decide receipts through the
//!   remaining typed buses.
//!
//! Only AIRs 0 and 1 expose public values, matching OpenVM's standard recursive
//! child interface. Consequently a completed proof can enter the existing
//! internal-for-leaf, internal-recursive, root, and Halo2 pipelines without a
//! History replay.

mod air;
mod bus;
mod circuit;
mod fresh_instance_link;
mod ordered_manifest;
mod prefix_verify;
mod public_values;
mod terminal_decide;
mod terminal_decide_complete;
mod terminal_rs_statement;
mod terminal_statement_prefix;
mod terminal_two_coset_folding;
mod terminal_two_coset_opening;
mod terminal_two_coset_query;
mod terminal_two_coset_transcript;
mod warp_verify;

pub use air::*;
pub use bus::*;
pub use circuit::*;
pub use fresh_instance_link::*;
pub use ordered_manifest::*;
pub use prefix_verify::*;
pub use public_values::*;
pub use terminal_decide::*;
pub use terminal_decide_complete::*;
pub use terminal_rs_statement::*;
pub use terminal_statement_prefix::*;
pub use terminal_two_coset_folding::*;
pub use terminal_two_coset_opening::*;
pub use terminal_two_coset_query::*;
pub use terminal_two_coset_transcript::*;
pub use warp_verify::*;

#[cfg(test)]
mod tests;

//! Row-oriented AIR primitives for certifying the same-root terminal WHIR
//! verifier attached to a native WARP accumulator.

/// Maximum degree of terminal schedule selector polynomials.
///
/// `Encoder` adds one degree for its validity constraints. Terminal verifier
/// schedules are public, fixed lookup tables, so keep their
// selector flags below the recursive history profile's degree-five ceiling:
// applying an `active` selector adds one degree to the encoded flag.
pub const NATIVE_TERMINAL_SELECTOR_MAX_FLAG_DEGREE: u32 = 4;

/// The adjoint sumcheck uses selector expressions inside a Lagrange-basis
/// recurrence. The selector is multiplied by the challenge, the prior prefix,
/// and the row activity flag, so degree two is the largest selector profile
/// compatible with the degree-five recursive history circuit.
pub const NATIVE_TERMINAL_SUMCHECK_SELECTOR_MAX_FLAG_DEGREE: u32 = 2;

mod adjoint_kernel;
mod adjoint_sumcheck;
mod bus;
mod expected;
mod final_check;
mod final_table;
mod fixed_multi_air;
mod fixed_multi_air_complete;
mod folding;
mod opened;
mod query;
mod round;
mod statement;
mod sumcheck;
mod terminal_two_coset_query;
mod weight;

pub use adjoint_kernel::*;
pub use adjoint_sumcheck::*;
pub use bus::*;
pub use expected::*;
pub use final_check::*;
pub use final_table::*;
pub use fixed_multi_air::*;
pub use fixed_multi_air_complete::*;
pub use folding::*;
pub use opened::*;
pub use query::*;
pub use round::*;
pub use statement::*;
pub use sumcheck::*;
pub use terminal_two_coset_query::*;
pub use weight::*;

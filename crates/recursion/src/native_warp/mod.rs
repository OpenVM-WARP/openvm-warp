//! AIR modules for certifying one native WARP VACC transition.

// Shared SWIRL-stacking endpoint tags. These are neutral transcript/bus tags;
// unlike the retired reduction module, they do not define a PESAT relation.
pub const NATIVE_REDUCTION_ENDPOINT_STACKING_POINT: usize = 3;
pub const NATIVE_REDUCTION_ENDPOINT_STACKING_OPENING: usize = 5;

pub mod base_fresh;
pub mod batching;
pub mod bus;
pub mod claim;
pub mod eq;
pub mod ext;
pub mod layout;
pub mod leaf_hash;
pub mod multiproof;
#[path = "product_opening/accumulator.rs"]
pub mod prior_accumulator;
pub mod reduced_swirl_source;
pub mod reduced_swirl_terminal;
pub mod reduced_swirl_vacc;
pub mod shift;
pub mod standard_vacc;
pub mod sumcheck;
pub mod terminal;
pub mod transcript;
pub mod twin;

pub use base_fresh::*;
pub use batching::*;
pub use bus::*;
pub use claim::*;
pub use eq::*;
pub use layout::*;
pub use leaf_hash::*;
pub use multiproof::*;
pub use prior_accumulator::*;
pub use reduced_swirl_source::*;
pub use reduced_swirl_terminal::*;
pub use reduced_swirl_vacc::*;
pub use shift::*;
pub use standard_vacc::*;
pub use sumcheck::*;
pub use terminal::*;
pub use transcript::*;
pub use twin::*;

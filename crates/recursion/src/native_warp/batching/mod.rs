mod air;
mod opening;

pub use air::*;
pub use opening::*;

pub const OPENING_SECTION_POINT: usize = 0;
pub const OPENING_SECTION_TARGET: usize = 1;
pub const BATCHING_OUTPUT_ALPHA: usize = 0;
pub const BATCHING_OUTPUT_MU: usize = 1;

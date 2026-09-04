mod challenge;
mod final_check;
mod fold;
mod sigma;

pub use final_check::*;
pub use fold::*;
pub use sigma::*;

pub const CLAIM_SECTION_ALPHA: usize = 0;
pub const CLAIM_SECTION_BETA: usize = 1;
pub const CLAIM_SECTION_MU: usize = 2;
pub const CLAIM_SECTION_ETA: usize = 3;
pub const TWIN_SCALAR_NU_0: usize = 0;
pub const TWIN_SCALAR_ETA: usize = 1;
pub use challenge::*;

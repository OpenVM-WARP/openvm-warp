mod air;
mod trace;
pub use air::*;
pub use trace::*;

/// Public so a circuit outside this crate can drive the GPU power checker directly. The native
/// WARP history certificate assembles three of the six verifier modules itself and therefore
/// supplies their primitives.
#[cfg(feature = "cuda")]
pub mod cuda;

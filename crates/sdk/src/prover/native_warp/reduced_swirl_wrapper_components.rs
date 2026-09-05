//! Common interface for reduced-SWIRL verifier components.
//!
//! The production transition leaf and terminal finalizer compose their AIR
//! inventories directly. Keeping only this interface avoids retaining the
//! superseded generic three-component wrapper runtime.

use openvm_stark_backend::{AirRef, StarkProtocolConfig};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, F};

/// One setup-fixed verifier component in the reduced-SWIRL proof pipeline.
///
/// Implementations own their buses, AIRs, and protocol-specific digest. The
/// digest covers every fixed dimension, relation/index identifier, transcript
/// version, and security parameter used by those AIRs.
pub trait ReducedSwirlVerifierComponent: Send + Sync + 'static {
    fn protocol_digest(&self) -> Digest;
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>>;
}

use std::sync::Arc;

use openvm_recursion_circuit::prelude::F;
use openvm_stark_backend::{AirRef, StarkProtocolConfig};
use openvm_stark_sdk::config::baby_bear_poseidon2::Digest;

use super::{
    FiniteWarpV3Binding, FiniteWarpV3Error, FiniteWarpV3ReceiptBuses, FiniteWarpV3StatementAir,
    FiniteWarpV3VerifierPvsAir, FiniteWarpV3VmPvsAir,
};
use crate::circuit::Circuit;

/// Receipt-producing verifier AIRs plugged into the fixed wrapper boundary.
///
/// A production implementation must verify the ordered manifest, every
/// ordinary WARP invocation, and terminal `Decide`.  Host preflight results
/// are not an implementation of this trait.  The component digest and AIR set
/// are bound by the final MultiSTARK verifying key.
pub trait FiniteWarpV3VerifierComponents: Send + Sync + 'static {
    fn receipt_buses(&self) -> FiniteWarpV3ReceiptBuses;
    fn component_digest(&self) -> Digest;
    fn component_air_count(&self) -> usize;
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>>;
}

/// Complete leaf-compatible circuit shape.  It deliberately has no
/// constructor that accepts host booleans standing in for component proofs.
pub struct FiniteWarpV3Circuit<C: FiniteWarpV3VerifierComponents> {
    pub binding: FiniteWarpV3Binding,
    pub verifier_pvs_air: Arc<FiniteWarpV3VerifierPvsAir>,
    pub vm_pvs_air: Arc<FiniteWarpV3VmPvsAir>,
    pub statement_air: Arc<FiniteWarpV3StatementAir>,
    pub components: Arc<C>,
}

impl<C: FiniteWarpV3VerifierComponents> FiniteWarpV3Circuit<C> {
    pub fn new(
        binding: FiniteWarpV3Binding,
        components: Arc<C>,
    ) -> Result<Self, FiniteWarpV3Error> {
        binding.validate()?;
        if components.component_air_count() == 0 {
            return Err(FiniteWarpV3Error::MissingVerifierComponents);
        }
        if components.component_digest() != binding.verifier_component_digest {
            return Err(FiniteWarpV3Error::VerifierComponentDigest);
        }
        let buses = components.receipt_buses();
        Ok(Self {
            verifier_pvs_air: Arc::new(FiniteWarpV3VerifierPvsAir::new(&binding)),
            vm_pvs_air: Arc::new(FiniteWarpV3VmPvsAir::new(buses.execution)),
            statement_air: Arc::new(FiniteWarpV3StatementAir::new(binding.clone(), buses)?),
            binding,
            components,
        })
    }
}

impl<SC, C> Circuit<SC> for FiniteWarpV3Circuit<C>
where
    SC: StarkProtocolConfig<F = F>,
    C: FiniteWarpV3VerifierComponents,
{
    fn airs(&self) -> Vec<AirRef<SC>> {
        let component_airs = self.components.airs::<SC>();
        assert_eq!(
            component_airs.len(),
            self.components.component_air_count(),
            "finite WARP v3 component AIR count changed after setup"
        );
        assert!(
            component_airs
                .iter()
                .all(|air| air.num_public_values() == 0),
            "finite WARP v3 receipt producers must expose zero public values"
        );
        [
            self.verifier_pvs_air.clone() as AirRef<SC>,
            self.vm_pvs_air.clone() as AirRef<SC>,
            self.statement_air.clone() as AirRef<SC>,
        ]
        .into_iter()
        .chain(component_airs)
        .collect()
    }
}

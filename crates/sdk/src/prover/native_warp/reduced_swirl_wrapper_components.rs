//! Composition of the three authenticated reduced-SWIRL wrapper components.
//!
//! Source verification, WARP Verify/VACC, and terminal Decide remain separate
//! logical relations. This module only gives them stable AIR-index ranges and
//! one setup-derived identity for the direct wrapper MultiSTARK.

use openvm_continuations::circuit::{
    reduced_swirl_source_receipt::ReducedSwirlSourceReceiptBlock,
    reduced_swirl_warp::{
        ReducedSwirlSourceReceiptMessage, ReducedSwirlWrapperBinding,
        ReducedSwirlWrapperReceiptBuses, ReducedSwirlWrapperRecord, ReducedSwirlWrapperStatement,
        ReducedSwirlWrapperVerifierComponents, REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION,
    },
};
use openvm_recursion_circuit::native_warp::{
    ReducedSwirlTerminalReceiptRecord, ReducedSwirlVaccChainReceiptMessage,
};
use openvm_stark_backend::{
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    prover::{AirProvingContext, ProverBackend},
    AirRef, FiatShamirTranscript, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, Digest, DIGEST_SIZE, F,
};

use super::reduced_swirl_wrapper_system::ReducedSwirlWrapperWitness;

const COMPONENT_DIGEST_TAG: &[u8] = b"openvm.native-warp.reduced-swirl.wrapper-components.v2";

/// Construct the single public wrapper statement from the three constrained
/// component receipts.
///
/// This host function is only a typed assembler. The wrapper statement AIR
/// independently consumes the source, VACC, terminal, and execution receipts
/// on distinct buses, so changing any field without changing its producer
/// leaves an unbalanced interaction.
pub fn reduced_swirl_wrapper_record_from_receipts(
    binding: &ReducedSwirlWrapperBinding,
    source: &ReducedSwirlSourceReceiptBlock,
    vacc: &ReducedSwirlVaccChainReceiptMessage<F>,
    terminal: &ReducedSwirlTerminalReceiptRecord,
) -> Result<ReducedSwirlWrapperRecord, &'static str> {
    let first = source.sources.first().ok_or("empty source receipt")?;
    let last = source.sources.last().ok_or("empty source receipt")?;
    let source_count = u32::try_from(source.sources.len()).map_err(|_| "source count")?;
    if source.source_offset != 0
        || first.segment_index != 0
        || last.segment_index + 1 != source_count
    {
        return Err("non-canonical full source receipt interval");
    }
    let source = ReducedSwirlSourceReceiptMessage {
        protocol_digest: binding.protocol_digest,
        manifest_digest: source.manifest_digest,
        source_offset: F::ZERO,
        source_count: F::from_u32(source_count),
        program_commitment: first.vm.program_commitment,
        initial_pc: first.vm.initial_pc,
        initial_root: first.vm.initial_root,
        final_pc: last.vm.final_pc,
        final_root: last.vm.final_root,
        exit_code: last.vm.exit_code,
        is_terminate: last.vm.is_terminate,
    };
    reduced_swirl_wrapper_record_from_receipt_messages(binding, &source, vacc, terminal)
}

/// Construct the wrapper record from the compact typed source-tree receipt.
///
/// This is the production seam for bounded source leaves: the receipt has
/// already been emitted by a complete one-child verifier and linked to the
/// detached VACC summary in-circuit.  The host only assembles the same values
/// into the public wrapper statement; it does not decide their validity.
pub fn reduced_swirl_wrapper_record_from_receipt_messages(
    binding: &ReducedSwirlWrapperBinding,
    source: &ReducedSwirlSourceReceiptMessage<F>,
    vacc: &ReducedSwirlVaccChainReceiptMessage<F>,
    terminal: &ReducedSwirlTerminalReceiptRecord,
) -> Result<ReducedSwirlWrapperRecord, &'static str> {
    binding.validate()?;
    let source_count = source.source_count.as_canonical_u32();
    let call_count = vacc.call_count.as_canonical_u32();
    if source_count == 0
        || source.source_offset != F::ZERO
        || source.protocol_digest != binding.protocol_digest
        || vacc.source_count != source.source_count
        || call_count == 0
        || source.manifest_digest != vacc.manifest_digest
        || vacc.protocol_digest != binding.protocol_digest
        || vacc.relation_digest != binding.relation_digest
        || vacc.warp_index_digest != binding.warp_index_digest
        || terminal.protocol_digest != binding.protocol_digest
        || terminal.relation_digest != binding.relation_digest
        || terminal.terminal_index_digest != binding.terminal_index_digest
        || terminal.verifier_component_digest != binding.verifier_component_digest
        || terminal.final_accumulator_digest != vacc.final_accumulator_digest
        || terminal.final_accumulator_root != vacc.final_accumulator_root
        || source.exit_code != F::ZERO
        || source.is_terminate != F::ONE
    {
        return Err("reduced-SWIRL receipt product mismatch");
    }
    let statement = ReducedSwirlWrapperStatement {
        protocol_version: binding.protocol_version,
        protocol_digest: binding.protocol_digest,
        relation_digest: binding.relation_digest,
        warp_index_digest: binding.warp_index_digest,
        terminal_index_digest: binding.terminal_index_digest,
        verifier_component_digest: binding.verifier_component_digest,
        schedule_digest: vacc.schedule_digest,
        manifest_digest: source.manifest_digest,
        source_count,
        call_count,
        program_commitment: source.program_commitment,
        initial_pc: source.initial_pc,
        initial_root: source.initial_root,
        final_pc: source.final_pc,
        final_root: source.final_root,
        final_accumulator_digest: vacc.final_accumulator_digest,
        final_accumulator_root: vacc.final_accumulator_root,
    };
    statement.validate(binding)?;
    Ok(ReducedSwirlWrapperRecord { statement })
}

/// One setup-fixed verifier component of the direct wrapper.
///
/// Implementations own their buses, AIRs, and protocol-specific digest. The
/// digest must cover every fixed dimension, relation/index identifier,
/// transcript version, and security parameter used by those AIRs.
pub trait ReducedSwirlVerifierComponent: Send + Sync + 'static {
    fn protocol_digest(&self) -> Digest;
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>>;
}

/// Direct product of the source, VACC, and terminal verifier relations.
pub struct ReducedSwirlProductionComponents<S, V, T> {
    buses: ReducedSwirlWrapperReceiptBuses,
    source: S,
    vacc: V,
    terminal: T,
    digest: Digest,
    counts: [usize; 3],
}

impl<S, V, T> ReducedSwirlProductionComponents<S, V, T>
where
    S: ReducedSwirlVerifierComponent,
    V: ReducedSwirlVerifierComponent,
    T: ReducedSwirlVerifierComponent,
{
    pub fn new(
        buses: ReducedSwirlWrapperReceiptBuses,
        source: S,
        vacc: V,
        terminal: T,
    ) -> Result<Self, &'static str> {
        let counts = [
            source.airs::<BabyBearPoseidon2Config>().len(),
            vacc.airs::<BabyBearPoseidon2Config>().len(),
            terminal.airs::<BabyBearPoseidon2Config>().len(),
        ];
        if counts.contains(&0) {
            return Err("empty reduced-SWIRL verifier component");
        }
        let digests = [
            source.protocol_digest(),
            vacc.protocol_digest(),
            terminal.protocol_digest(),
        ];
        if digests
            .iter()
            .any(|digest| digest.iter().all(|value| *value == F::ZERO))
        {
            return Err("unset reduced-SWIRL component digest");
        }
        let digest = reduced_swirl_wrapper_component_digest(buses, digests, counts)?;
        Ok(Self {
            buses,
            source,
            vacc,
            terminal,
            digest,
            counts,
        })
    }

    #[must_use]
    pub const fn source(&self) -> &S {
        &self.source
    }

    #[must_use]
    pub const fn vacc(&self) -> &V {
        &self.vacc
    }

    #[must_use]
    pub const fn terminal(&self) -> &T {
        &self.terminal
    }

    #[must_use]
    pub const fn component_counts(&self) -> [usize; 3] {
        self.counts
    }

    /// Offset component-local contexts into the wrapper's component AIR
    /// namespace. Every local AIR must occur exactly once; active-prefix AIRs
    /// carry runtime multiplicity in rows, not by duplicating keyed AIR IDs.
    pub fn assemble_witness<PB: ProverBackend<Val = F>>(
        &self,
        source: Vec<AirProvingContext<PB>>,
        vacc: Vec<AirProvingContext<PB>>,
        terminal: Vec<AirProvingContext<PB>>,
    ) -> Result<ReducedSwirlWrapperWitness<PB>, &'static str> {
        for (actual, expected) in [source.len(), vacc.len(), terminal.len()]
            .into_iter()
            .zip(self.counts)
        {
            if actual != expected {
                return Err("reduced-SWIRL component witness count");
            }
        }
        let source_offset = 0;
        let vacc_offset = self.counts[0];
        let terminal_offset = vacc_offset + self.counts[1];
        let component_contexts = source
            .into_iter()
            .enumerate()
            .map(|(index, context)| (source_offset + index, context))
            .chain(
                vacc.into_iter()
                    .enumerate()
                    .map(|(index, context)| (vacc_offset + index, context)),
            )
            .chain(
                terminal
                    .into_iter()
                    .enumerate()
                    .map(|(index, context)| (terminal_offset + index, context)),
            )
            .collect();
        Ok(ReducedSwirlWrapperWitness { component_contexts })
    }
}

impl<S, V, T> ReducedSwirlWrapperVerifierComponents for ReducedSwirlProductionComponents<S, V, T>
where
    S: ReducedSwirlVerifierComponent,
    V: ReducedSwirlVerifierComponent,
    T: ReducedSwirlVerifierComponent,
{
    fn receipt_buses(&self) -> ReducedSwirlWrapperReceiptBuses {
        self.buses
    }

    fn component_digest(&self) -> Digest {
        self.digest
    }

    fn component_air_count(&self) -> usize {
        self.counts.iter().sum()
    }

    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        self.source
            .airs::<SC>()
            .into_iter()
            .chain(self.vacc.airs::<SC>())
            .chain(self.terminal.airs::<SC>())
            .collect()
    }
}

/// Setup identity for the direct product verifier. Bus numbers are included
/// because they determine interaction wiring; the wrapper VK independently
/// binds the resulting AIR inventory as a second line of defense.
pub fn reduced_swirl_wrapper_component_digest(
    buses: ReducedSwirlWrapperReceiptBuses,
    digests: [Digest; 3],
    counts: [usize; 3],
) -> Result<Digest, &'static str> {
    if counts.contains(&0)
        || digests
            .iter()
            .any(|digest| digest.iter().all(|value| *value == F::ZERO))
    {
        return Err("invalid reduced-SWIRL component material");
    }
    let mut transcript = default_duplex_sponge_recorder();
    observe_bytes(&mut transcript, COMPONENT_DIGEST_TAG);
    observe_u64(
        &mut transcript,
        u64::from(REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION),
    );
    for digest in digests {
        for value in digest {
            <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(&mut transcript, value);
        }
    }
    for count in counts {
        observe_u64(
            &mut transcript,
            u64::try_from(count).map_err(|_| "component AIR count")?,
        );
    }
    for bus in [
        buses.source.index(),
        buses.vacc.index(),
        buses.terminal.index(),
        buses.execution.index(),
    ] {
        observe_u64(&mut transcript, u64::from(bus));
    }
    Ok(core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    }))
}

fn observe_bytes(
    transcript: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>,
    bytes: &[u8],
) {
    observe_u64(transcript, bytes.len() as u64);
    for &byte in bytes {
        transcript.observe(F::from_u8(byte));
    }
}

fn observe_u64(transcript: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>, value: u64) {
    for byte in value.to_le_bytes() {
        transcript.observe(F::from_u8(byte));
    }
}

const _: [(); DIGEST_SIZE] = [(); 8];

#[cfg(test)]
mod tests {
    use openvm_stark_backend::interaction::BusIndex;

    use super::*;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
    }

    #[test]
    fn component_identity_binds_order_counts_and_buses() {
        let buses = ReducedSwirlWrapperReceiptBuses::new(900 as BusIndex);
        let baseline = reduced_swirl_wrapper_component_digest(
            buses,
            [digest(1), digest(20), digest(40)],
            [3, 5, 7],
        )
        .unwrap();
        assert_ne!(
            baseline,
            reduced_swirl_wrapper_component_digest(
                buses,
                [digest(20), digest(1), digest(40)],
                [3, 5, 7],
            )
            .unwrap()
        );
        assert_ne!(
            baseline,
            reduced_swirl_wrapper_component_digest(
                ReducedSwirlWrapperReceiptBuses::new(904 as BusIndex),
                [digest(1), digest(20), digest(40)],
                [3, 5, 7],
            )
            .unwrap()
        );
        assert_ne!(
            baseline,
            reduced_swirl_wrapper_component_digest(
                buses,
                [digest(1), digest(20), digest(40)],
                [3, 6, 7],
            )
            .unwrap()
        );
    }
}

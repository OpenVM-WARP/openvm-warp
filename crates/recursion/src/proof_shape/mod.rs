use core::cmp::Reverse;
use std::sync::Arc;

use itertools::{izip, Itertools};
use openvm_circuit_primitives::encoder::Encoder;
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{
    keygen::types::{MultiStarkVerifyingKey, VerifierSinglePreprocessedData},
    proof::Proof,
    prover::AirProvingContext,
    AirRef, FiatShamirTranscript, StarkProtocolConfig, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    primitives::{
        bus::{PowerCheckerBus, RangeCheckerBus},
        pow::PowerCheckerCpuTraceGenerator,
        range::{RangeCheckerAir, RangeCheckerCpuTraceGenerator},
    },
    proof_shape::{
        bus::{
            NumPublicValuesBus, ProofShapeMetadataBus, ProofShapePermutationBus, StartingTidxBus,
        },
        proof_shape::{generate_metadata_dummy_trace, ProofShapeAir, ProofShapeMetadataAir},
        pvs::PublicValuesAir,
    },
    system::{
        frame::MultiStarkVkeyFrame, AirModule, BusIndexManager, BusInventory, GlobalCtxCpu,
        Preflight, ProofShapePreflight, TraceGenModule, POW_CHECKER_HEIGHT,
    },
    tracegen::{ModuleChip, RowMajorChip},
};

pub mod bus;
#[allow(clippy::module_inception)]
pub mod proof_shape;
pub mod pvs;

#[cfg(feature = "cuda")]
mod cuda_abi;

#[derive(Clone)]
pub struct AirMetadata {
    pub(crate) is_required: bool,
    pub(crate) need_rot: bool,
    pub(crate) num_public_values: usize,
    pub(crate) num_interactions: usize,
    pub(crate) main_width: usize,
    pub(crate) cached_widths: Vec<usize>,
    pub(crate) preprocessed_width: Option<usize>,
    pub(crate) preprocessed_data: Option<VerifierSinglePreprocessedData<Digest>>,
}

pub struct ProofShapeModule {
    // Verifying key fields
    per_air: Vec<AirMetadata>,
    l_skip: usize,
    /// Threshold from the child VK used by [`ProofShapeAir`] on the summary row:
    /// `sum_i(num_interactions[i] * lifted_height[i]) < max_interaction_count`,
    /// with `lifted_height[i] = max(trace_height[i], 2^l_skip)`.
    max_interaction_count: u32,
    air_idx_gap_bits: usize,

    // Buses (inventory for external, others are internal)
    bus_inventory: BusInventory,
    range_bus: RangeCheckerBus,
    pow_bus: PowerCheckerBus,
    permutation_bus: ProofShapePermutationBus,
    metadata_bus: Option<ProofShapeMetadataBus>,
    starting_tidx_bus: StartingTidxBus,
    num_pvs_bus: NumPublicValuesBus,

    // Required for ProofShapeAir tracegen + constraints
    idx_encoder: Arc<Encoder>,
    min_cached_idx: usize,
    max_cached: usize,
    commit_mult: usize,
    /// Additional setup-fixed consumer of `(AIR id, presence, height)` used
    /// by the ordered deferred-SWIRL source receipt.
    layout_export_lookups: usize,

    // Module sends extra public values message for use outside of verifier
    // sub-circuit if true
    continuations_enabled: bool,
}

/// Above this child-AIR count, a one-proof verifier with no cached child mains
/// authenticates VK metadata through a committed lookup table.  Smaller and
/// multi-proof recursive circuits retain the existing degree-two encoder: for
/// those keys its lower interaction arity is cheaper than another AIR.
pub const PROOF_SHAPE_METADATA_LOOKUP_THRESHOLD: usize = 256;

/// Largest child verifying key the verifier sub-circuit can read.
///
/// [`ProofShapeAir`] sorts rows by descending height then ascending AIR index, and range-checks
/// the *gap* between consecutive equal-height indices against the selected 10- or 12-bit bus.
/// The 4096-AIR cap is a conservative sufficient condition for the widest bus; keys whose
/// equal-height runs are dense can have much smaller actual gaps.
///
/// Raised to 4096 for the native WARP bounded history span. The heavy Reth profile keys roughly
/// 3256 AIR instances; fitting them in one MultiSTARK removes the repeated recursive history
/// verifier. Large one-proof keys do not pay a 4096-way symbolic selector: they use the
/// VK-committed metadata table selected by [`PROOF_SHAPE_METADATA_LOOKUP_THRESHOLD`].
///
/// The gap bound is the real constraint and it does *not* come for free at this size: a 737-AIR
/// span child can produce a gap close to the full child-AIR range, so
/// [`RECURSION_AIR_IDX_GAP_BITS`] went to twelve alongside this. `air_idx_gap` in
/// `proof_shape/trace.rs` enforces it explicitly, so exceeding it names
/// the offending indices rather than producing an unprovable circuit.
///
/// Callers that assemble a child circuit check their AIR count against this before keying, so
/// overflowing it is a reported fallback rather than a panic.
pub const RECURSION_MAX_CHILD_AIRS: usize = 1 << 12;

/// Bit width of [`ProofShapeAir`]'s AIR-index gap range check.
///
/// Eight in the upstream verifier, ten for the first 1024-AIR bounded profile, and twelve for
/// the single-proof 4096-AIR heavy-block profile. Sparse equal-height runs can span almost
/// the complete child key, so the gap width must track the accepted key envelope.
///
/// This width has its own [`RangeCheckerAir`] because the range bus is keyed `(value, max_bits)`
/// and a table publishes only its own width: the existing `<8>` table serves the `LIMB_BITS`
/// decompositions and cannot also answer the larger-width query. `PowerCheckerAir` is the same
/// bus's third provider, at width 5.
pub const RECURSION_AIR_IDX_GAP_BITS: usize = 12;

impl ProofShapeModule {
    pub fn new(
        mvk: &MultiStarkVkeyFrame,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        continuations_enabled: bool,
        max_num_proofs: usize,
    ) -> Self {
        assert!(
            mvk.per_air.len() <= RECURSION_MAX_CHILD_AIRS,
            "recursion circuit only supports child verifying keys with at most {RECURSION_MAX_CHILD_AIRS} AIRs"
        );

        let idx_encoder = Arc::new(Encoder::new(mvk.per_air.len(), 2, true));

        let (min_cached_idx, min_cached) = mvk
            .per_air
            .iter()
            .enumerate()
            .min_by_key(|(_, avk)| avk.params.width.cached_mains.len())
            .map(|(idx, avk)| (idx, avk.params.width.cached_mains.len()))
            .unwrap();
        let mut max_cached = mvk
            .per_air
            .iter()
            .map(|avk| avk.params.width.cached_mains.len())
            .max()
            .unwrap();
        if min_cached == max_cached {
            max_cached += 1;
        }

        let per_air = mvk
            .per_air
            .iter()
            .map(|avk| AirMetadata {
                is_required: avk.is_required,
                need_rot: avk.params.need_rot,
                num_public_values: avk.params.num_public_values,
                num_interactions: avk.num_interactions,
                main_width: avk.params.width.common_main,
                cached_widths: avk.params.width.cached_mains.clone(),
                preprocessed_width: avk.params.width.preprocessed,
                preprocessed_data: avk.preprocessed_data.clone(),
            })
            .collect_vec();

        // One table row is consumed exactly once, so this representation is
        // intentionally limited to verifier circuits that prove exactly one
        // child.  Cached mains require a variable-width tuple and remain on
        // the established encoder path until a concrete large-key user needs
        // them.  The WARP normalization child has no cached mains.
        let metadata_lookup_enabled = max_num_proofs == 1
            && per_air.len() >= PROOF_SHAPE_METADATA_LOOKUP_THRESHOLD
            && per_air
                .iter()
                .all(|metadata| metadata.cached_widths.is_empty());
        let air_idx_gap_bits = if per_air.len() <= 1 << 10 { 10 } else { 12 };

        let range_bus = bus_inventory.range_checker_bus;
        let pow_bus = bus_inventory.power_checker_bus;
        Self {
            per_air,
            l_skip: mvk.params.l_skip,
            max_interaction_count: mvk.params.logup.max_interaction_count,
            air_idx_gap_bits,
            bus_inventory,
            range_bus,
            pow_bus,
            permutation_bus: ProofShapePermutationBus::new(b.new_bus_idx()),
            metadata_bus: metadata_lookup_enabled
                .then(|| ProofShapeMetadataBus::new(b.new_bus_idx())),
            starting_tidx_bus: StartingTidxBus::new(b.new_bus_idx()),
            num_pvs_bus: NumPublicValuesBus::new(b.new_bus_idx()),
            idx_encoder,
            min_cached_idx,
            max_cached,
            commit_mult: mvk.params.whir.rounds.first().unwrap().num_queries,
            layout_export_lookups: 0,
            continuations_enabled,
        }
    }

    /// Configure how many consumers authenticate each commitment when WHIR
    /// is deferred to an enclosing reduced-SWIRL WARP proof.
    pub fn set_deferred_whir_commitment_multiplicity(&mut self, commit_mult: usize) {
        self.commit_mult = commit_mult;
    }

    /// Publish one additional authenticated proof-shape view to an enclosing
    /// receipt AIR. This changes only lookup multiplicities; trace generation
    /// and the child transcript remain byte-for-byte unchanged.
    pub fn set_layout_export_lookups(&mut self, lookups: usize) {
        self.layout_export_lookups = lookups;
    }

    #[tracing::instrument(level = "trace", skip_all)]
    pub fn run_preflight<TS>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        ts: &mut TS,
    ) where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config> + TranscriptHistory,
    {
        ts.observe_commit(child_vk.pre_hash);
        ts.observe_commit(proof.common_main_commit);

        let mut pvs_tidx = vec![];
        let mut starting_tidx = vec![];

        for (trace_vdata, avk, pvs) in izip!(
            &proof.trace_vdata,
            &child_vk.inner.per_air,
            &proof.public_values
        ) {
            let is_air_present = trace_vdata.is_some();
            starting_tidx.push(ts.len());

            if !avk.is_required {
                ts.observe(F::from_bool(is_air_present));
            }
            if let Some(trace_vdata) = trace_vdata {
                if let Some(pdata) = avk.preprocessed_data.as_ref() {
                    ts.observe_commit(pdata.commit);
                } else {
                    ts.observe(F::from_usize(trace_vdata.log_height));
                }
                debug_assert_eq!(avk.num_cached_mains(), trace_vdata.cached_commitments.len());
                if !pvs.is_empty() {
                    pvs_tidx.push(ts.len());
                }
                for commit in &trace_vdata.cached_commitments {
                    ts.observe_commit(*commit);
                }
                debug_assert_eq!(avk.params.num_public_values, pvs.len());
            }
            for pv in pvs {
                ts.observe(*pv);
            }
        }

        self.finish_preflight(
            child_vk,
            proof,
            preflight,
            starting_tidx,
            pvs_tidx,
            ts.len(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_preflight(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        starting_tidx: Vec<usize>,
        pvs_tidx: Vec<usize>,
        post_tidx: usize,
    ) {
        let l_skip = child_vk.inner.params.l_skip;
        let mut sorted_trace_vdata: Vec<_> = proof
            .trace_vdata
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(air_id, data)| data.map(|data| (air_id, data)))
            .collect();
        sorted_trace_vdata.sort_by_key(|(air_idx, data)| (Reverse(data.log_height), *air_idx));

        let n_max = proof
            .trace_vdata
            .iter()
            .flat_map(|datum| {
                datum
                    .as_ref()
                    .map(|datum| datum.log_height.saturating_sub(l_skip))
            })
            .max()
            .unwrap();
        let num_layers = proof.gkr_proof.claims_per_layer.len();
        let n_logup = num_layers.saturating_sub(l_skip);

        preflight.proof_shape = ProofShapePreflight {
            sorted_trace_vdata,
            starting_tidx,
            pvs_tidx,
            post_tidx,
            n_max,
            n_logup,
            l_skip: child_vk.inner.params.l_skip,
        };
    }
}

impl AirModule for ProofShapeModule {
    fn num_airs(&self) -> usize {
        // ProofShape, PublicValues, and one RangeChecker per width the AIR queries: `<8>` for the
        // `LIMB_BITS` decompositions and `<RECURSION_AIR_IDX_GAP_BITS>` for the AIR-index gap.
        4 + usize::from(self.metadata_bus.is_some())
    }

    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let proof_shape_air = ProofShapeAir::<4, 8> {
            per_air: self.per_air.clone(),
            l_skip: self.l_skip,
            min_cached_idx: self.min_cached_idx,
            max_cached: self.max_cached,
            commit_mult: self.commit_mult,
            max_interaction_count: self.max_interaction_count,
            air_idx_gap_bits: self.air_idx_gap_bits,
            idx_encoder: self.idx_encoder.clone(),
            metadata_bus: self.metadata_bus,
            range_bus: self.range_bus,
            pow_bus: self.pow_bus,
            permutation_bus: self.permutation_bus,
            starting_tidx_bus: self.starting_tidx_bus,
            num_pvs_bus: self.num_pvs_bus,
            fraction_folder_input_bus: self.bus_inventory.fraction_folder_input_bus,
            expression_claim_n_max_bus: self.bus_inventory.expression_claim_n_max_bus,
            gkr_module_bus: self.bus_inventory.gkr_module_bus,
            air_shape_bus: self.bus_inventory.air_shape_bus,
            air_presence_bus: self.bus_inventory.air_presence_bus,
            hyperdim_bus: self.bus_inventory.hyperdim_bus,
            lifted_heights_bus: self.bus_inventory.lifted_heights_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            n_lift_bus: self.bus_inventory.n_lift_bus,
            eq_n_logup_n_max_bus: self.bus_inventory.eq_n_logup_n_max_bus,
            eq_3b_shape_bus: self.bus_inventory.eq_3b_shape_bus,
            cached_commit_bus: self.bus_inventory.cached_commit_bus,
            pre_hash_bus: self.bus_inventory.pre_hash_bus,
            continuations_enabled: self.continuations_enabled,
            layout_export_lookups: self.layout_export_lookups,
        };
        let pvs_air = PublicValuesAir {
            public_values_bus: self.bus_inventory.public_values_bus,
            num_pvs_bus: self.num_pvs_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            continuations_enabled: self.continuations_enabled,
        };
        let range_checker = RangeCheckerAir::<8> {
            bus: self.range_bus,
        };
        let mut airs = vec![
            Arc::new(proof_shape_air) as AirRef<_>,
            Arc::new(pvs_air) as AirRef<_>,
            Arc::new(range_checker) as AirRef<_>,
        ];
        match self.air_idx_gap_bits {
            10 => airs.push(Arc::new(RangeCheckerAir::<10> {
                bus: self.range_bus,
            }) as AirRef<_>),
            12 => airs.push(Arc::new(RangeCheckerAir::<12> {
                bus: self.range_bus,
            }) as AirRef<_>),
            bits => panic!("unsupported proof-shape AIR-index gap width {bits}"),
        }
        if let Some(bus) = self.metadata_bus {
            airs.push(Arc::new(ProofShapeMetadataAir {
                per_air: self.per_air.clone(),
                l_skip: self.l_skip,
                min_cached_idx: self.min_cached_idx,
                bus,
            }) as AirRef<_>);
        }
        airs
    }
}

impl<SC: StarkProtocolConfig<F = F>> TraceGenModule<GlobalCtxCpu, CpuBackend<SC>>
    for ProofShapeModule
{
    // (pow_checker, external_range_checks)
    type ModuleSpecificCtx<'a> = (
        Arc<PowerCheckerCpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
        &'a [usize],
    );

    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        ctx: &Self::ModuleSpecificCtx<'_>,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<SC>>>> {
        let pow_checker = &ctx.0;
        let external_range_checks = ctx.1;

        let range_checker = Arc::new(RangeCheckerCpuTraceGenerator::<8>::default());
        let gap_range_checker =
            proof_shape::ProofShapeGapRangeCheckerCpu::new(self.air_idx_gap_bits);
        let proof_shape = proof_shape::ProofShapeChip::<4, 8>::new(
            self.idx_encoder.clone(),
            self.metadata_bus
                .map(|_| Arc::<[AirMetadata]>::from(self.per_air.clone())),
            self.min_cached_idx,
            self.max_cached,
            range_checker.clone(),
            gap_range_checker.clone(),
            pow_checker.clone(),
        );
        let ctx = (child_vk, proofs, preflights);
        let chips = [
            ProofShapeModuleChip::ProofShape(proof_shape),
            ProofShapeModuleChip::PublicValues,
        ];
        let mut ctxs: Vec<_> = chips
            .par_iter()
            .map(|chip| {
                chip.generate_proving_ctx(
                    &ctx,
                    required_heights.map(|heights| heights[chip.index()]),
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Option<Vec<_>>>()?;

        for &val in external_range_checks {
            range_checker.add_count(val);
        }
        tracing::trace_span!("wrapper.generate_trace", air = "RangeChecker").in_scope(|| {
            ctxs.push(AirProvingContext::simple_no_pis(
                range_checker.generate_trace_row_major(),
            ));
            ctxs.push(AirProvingContext::simple_no_pis(
                gap_range_checker.generate_trace_row_major(),
            ));
        });
        if self.metadata_bus.is_some() {
            let required_height = required_heights.map(|heights| heights[4]);
            let height = required_height.unwrap_or_else(|| self.per_air.len().next_power_of_two());
            if height < self.per_air.len() {
                return None;
            }
            ctxs.push(AirProvingContext::simple_no_pis(
                generate_metadata_dummy_trace(height),
            ));
        }
        Some(ctxs)
    }
}

#[derive(strum_macros::Display, strum::EnumDiscriminants)]
#[strum_discriminants(repr(usize))]
enum ProofShapeModuleChip {
    ProofShape(proof_shape::ProofShapeChip<4, 8>),
    PublicValues,
}

impl ProofShapeModuleChip {
    fn index(&self) -> usize {
        ProofShapeModuleChipDiscriminants::from(self) as usize
    }
}

impl RowMajorChip<F> for ProofShapeModuleChip {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a [Proof<BabyBearPoseidon2Config>],
        &'a [Preflight],
    );

    #[tracing::instrument(
        name = "wrapper.generate_trace",
        level = "trace",
        skip_all,
        fields(air = %self)
    )]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        use ProofShapeModuleChip::*;
        match self {
            ProofShape(chip) => chip.generate_trace(ctx, required_height),
            PublicValues => {
                pvs::PublicValuesTraceGenerator.generate_trace(&(ctx.1, ctx.2), required_height)
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use openvm_cuda_backend::GpuBackend;

    use super::*;
    use crate::{
        cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu, GlobalCtxGpu},
        primitives::{
            pow::cuda::PowerCheckerGpuTraceGenerator, range::cuda::RangeCheckerGpuTraceGenerator,
        },
    };

    impl TraceGenModule<GlobalCtxGpu, GpuBackend> for ProofShapeModule {
        type ModuleSpecificCtx<'a> = (
            Arc<PowerCheckerGpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
            &'a [usize],
            &'a openvm_cuda_common::stream::GpuDeviceCtx,
        );

        #[tracing::instrument(skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            ctx: &Self::ModuleSpecificCtx<'_>,
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContext<GpuBackend>>> {
            use crate::tracegen::ModuleChip;

            let pow_checker_gpu = &ctx.0;
            let external_range_checks = ctx.1;
            let device_ctx = ctx.2;

            let range_checker_gpu = Arc::new(RangeCheckerGpuTraceGenerator::<8>::from_vals(
                external_range_checks,
                device_ctx.clone(),
            ));
            let gap_range_checker_gpu = proof_shape::cuda::ProofShapeGapRangeCheckerGpu::new(
                self.air_idx_gap_bits,
                device_ctx.clone(),
            );
            let proof_shape_chip = proof_shape::cuda::ProofShapeChipGpu::<4, 8>::new(
                self.idx_encoder.width(),
                self.metadata_bus.is_some(),
                self.min_cached_idx,
                self.max_cached,
                range_checker_gpu.clone(),
                gap_range_checker_gpu.clone(),
                pow_checker_gpu.clone(),
            );
            let mut ctxs = Vec::with_capacity(4);
            // PERF[jpw]: we avoid par_iter so that kernel launches occur on the same stream.
            // This can be parallelized to separate streams for more CUDA stream parallelism, but it
            // will require recording events so streams properly sync for cudaMemcpyAsync and kernel
            // launches
            let proof_shape_ctx =
                tracing::trace_span!("wrapper.generate_trace", air = "ProofShape").in_scope(
                    || {
                        proof_shape_chip.generate_proving_ctx(
                            &(child_vk, preflights, device_ctx),
                            required_heights.map(|heights| heights[0]),
                        )
                    },
                )?;
            ctxs.push(proof_shape_ctx);

            let public_values_ctx =
                tracing::trace_span!("wrapper.generate_trace", air = "PublicValues").in_scope(
                    || {
                        pvs::cuda::PublicValuesGpuTraceGenerator.generate_proving_ctx(
                            &(proofs, preflights, device_ctx),
                            required_heights.map(|heights| heights[1]),
                        )
                    },
                )?;
            ctxs.push(public_values_ctx);
            // Drop the proof_shape chip so we can finalize auxiliary trace state (it holds Arc
            // clones).
            drop(proof_shape_chip);
            // Caution: proof_shape **must** finish trace gen before we materialize range checker
            // trace or sync power checker multiplicities to CPU.
            tracing::trace_span!("wrapper.generate_trace", air = "RangeChecker").in_scope(|| {
                ctxs.push(AirProvingContext::simple_no_pis(
                    Arc::try_unwrap(range_checker_gpu)
                        .ok()
                        .expect("range checker still shared")
                        .generate_trace(),
                ));
                ctxs.push(AirProvingContext::simple_no_pis(
                    gap_range_checker_gpu
                        .into_trace()
                        .expect("gap range checker still shared"),
                ));
            });

            if self.metadata_bus.is_some() {
                let required_height = required_heights.map(|heights| heights[4]);
                let height = required_height
                    .unwrap_or_else(|| self.per_air.len().max(1).next_power_of_two());
                if height < self.per_air.len() {
                    return None;
                }
                let trace = openvm_cuda_backend::base::DeviceMatrix::<F>::with_capacity_on(
                    height, 1, device_ctx,
                );
                trace.buffer().fill_zero_on(device_ctx).ok()?;
                ctxs.push(AirProvingContext::simple_no_pis(trace));
            }

            Some(ctxs)
        }
    }
}

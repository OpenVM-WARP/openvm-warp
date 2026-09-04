use std::sync::Arc;

use itertools::Itertools;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::Digest, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, memory_manager::MemTracker, stream::GpuDeviceCtx};
use openvm_stark_backend::prover::AirProvingContext;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    cuda::{preflight::PreflightGpu, vk::VerifyingKeyGpu},
    primitives::{
        pow::cuda::PowerCheckerGpuTraceGenerator, range::cuda::RangeCheckerGpuTraceGenerator,
    },
    proof_shape::{
        cuda_abi::proof_shape_tracegen,
        proof_shape::{ProofShapeCols, ProofShapeMetadataCols},
    },
    system::POW_CHECKER_HEIGHT,
    tracegen::ModuleChip,
};

#[repr(C)]
pub(crate) struct ProofShapePerProof {
    num_present: usize,
    n_max: usize,
    n_logup: usize,
    final_cidx: usize,
    final_total_interactions: usize,
    main_commit: Digest,
}

#[repr(C)]
pub(crate) struct ProofShapeTracegenInputs {
    num_airs: usize,
    l_skip: usize,
    max_interaction_count: u32,
    max_cached: usize,
    min_cached_idx: usize,
    selector_width: usize,
    metadata_lookup: u32,
    air_idx_gap_bits: usize,
    pre_hash: Digest,
    range_checker_8_ptr: *mut u32,
    range_checker_5_ptr: *mut u32,
    /// Table for the AIR-index gap, checked at the selected width rather than `LIMB_BITS`.
    /// Separate because the range bus is keyed `(value, max_bits)`.
    range_checker_gap_ptr: *mut u32,
    pow_checker_ptr: *mut u32,
}

#[derive(Clone)]
pub(in crate::proof_shape) enum ProofShapeGapRangeCheckerGpu {
    Bits10(Arc<RangeCheckerGpuTraceGenerator<10>>),
    Bits12(Arc<RangeCheckerGpuTraceGenerator<12>>),
}

impl ProofShapeGapRangeCheckerGpu {
    pub fn new(bits: usize, device_ctx: GpuDeviceCtx) -> Self {
        match bits {
            10 => Self::Bits10(Arc::new(RangeCheckerGpuTraceGenerator::new(device_ctx))),
            12 => Self::Bits12(Arc::new(RangeCheckerGpuTraceGenerator::new(device_ctx))),
            _ => panic!("unsupported proof-shape AIR-index gap width {bits}"),
        }
    }

    fn bits(&self) -> usize {
        match self {
            Self::Bits10(_) => 10,
            Self::Bits12(_) => 12,
        }
    }

    fn count_mut_ptr(&self) -> *mut u32 {
        match self {
            Self::Bits10(generator) => generator.count_mut_ptr(),
            Self::Bits12(generator) => generator.count_mut_ptr(),
        }
    }

    pub fn into_trace(self) -> Option<DeviceMatrix<openvm_cuda_backend::prelude::F>> {
        match self {
            Self::Bits10(generator) => Arc::try_unwrap(generator).ok().map(|g| g.generate_trace()),
            Self::Bits12(generator) => Arc::try_unwrap(generator).ok().map(|g| g.generate_trace()),
        }
    }
}

#[derive(derive_new::new)]
pub(in crate::proof_shape) struct ProofShapeChipGpu<const NUM_LIMBS: usize, const LIMB_BITS: usize>
{
    encoder_width: usize,
    metadata_lookup: bool,
    min_cached_idx: usize,
    max_cached: usize,
    range_checker: Arc<RangeCheckerGpuTraceGenerator<LIMB_BITS>>,
    gap_range_checker: ProofShapeGapRangeCheckerGpu,
    pow_checker: Arc<PowerCheckerGpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
}

const NUM_LIMBS: usize = 4;
const LIMB_BITS: usize = 8;
impl ModuleChip<GpuBackend> for ProofShapeChipGpu<NUM_LIMBS, LIMB_BITS> {
    type Ctx<'a> = (&'a VerifyingKeyGpu, &'a [PreflightGpu], &'a GpuDeviceCtx);

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_proving_ctx(
        &self,
        ctx: &Self::Ctx<'_>,
        height: Option<usize>,
    ) -> Option<AirProvingContext<GpuBackend>> {
        let (vk_gpu, preflights_gpu, device_ctx) = ctx;
        let mem = MemTracker::start("tracegen.proof_shape");
        let num_valid_rows = preflights_gpu.len() * (vk_gpu.per_air.len() + 1);
        let height = if let Some(height) = height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };
        let selector_width = if self.metadata_lookup {
            ProofShapeMetadataCols::<u8>::width()
        } else {
            self.encoder_width
        };
        let min_cached_idx = self.min_cached_idx;
        let max_cached = self.max_cached;
        let range_checker = &self.range_checker;
        let pow_checker = &self.pow_checker;
        let num_airs = vk_gpu.per_air.len();
        let width =
            ProofShapeCols::<u8, NUM_LIMBS>::width() + selector_width + max_cached * DIGEST_SIZE;
        let trace = DeviceMatrix::with_capacity_on(height, width, device_ctx);

        let per_row_tidx = preflights_gpu
            .iter()
            .map(|preflight| preflight.proof_shape.per_row_tidx.as_ptr())
            .collect_vec();
        let sorted_trace_heights = preflights_gpu
            .iter()
            .map(|preflight| preflight.proof_shape.sorted_trace_heights.as_ptr())
            .collect_vec();
        let sorted_trace_metadata = preflights_gpu
            .iter()
            .map(|preflight| preflight.proof_shape.sorted_trace_metadata.as_ptr())
            .collect_vec();
        let cached_commits = preflights_gpu
            .iter()
            .map(|preflight| preflight.proof_shape.sorted_cached_commits.as_ptr())
            .collect_vec();
        let per_proof = preflights_gpu
            .iter()
            .map(|preflight| ProofShapePerProof {
                num_present: preflight.proof_shape.num_present,
                n_max: preflight.proof_shape.n_max,
                n_logup: preflight.proof_shape.n_logup,
                final_cidx: preflight.proof_shape.final_cidx,
                final_total_interactions: preflight.proof_shape.final_total_interactions,
                main_commit: preflight.proof_shape.main_commit,
            })
            .collect_vec()
            .to_device_on(device_ctx)
            .unwrap();
        let inputs = ProofShapeTracegenInputs {
            num_airs,
            l_skip: vk_gpu.system_params.l_skip,
            max_interaction_count: vk_gpu.system_params.logup.max_interaction_count,
            max_cached,
            min_cached_idx,
            selector_width,
            metadata_lookup: u32::from(self.metadata_lookup),
            air_idx_gap_bits: self.gap_range_checker.bits(),
            pre_hash: vk_gpu.pre_hash,
            range_checker_8_ptr: range_checker.count_mut_ptr(),
            range_checker_5_ptr: pow_checker.range_count_mut_ptr(),
            range_checker_gap_ptr: self.gap_range_checker.count_mut_ptr(),
            pow_checker_ptr: pow_checker.pow_count_mut_ptr(),
        };

        unsafe {
            proof_shape_tracegen(
                trace.buffer(),
                height,
                &vk_gpu.per_air,
                per_row_tidx,
                sorted_trace_heights,
                sorted_trace_metadata,
                cached_commits,
                &per_proof,
                preflights_gpu.len(),
                &inputs,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        mem.emit_metrics();
        Some(AirProvingContext::simple_no_pis(trace))
    }
}

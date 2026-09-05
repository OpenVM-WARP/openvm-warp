//! Shared CPU-to-CUDA transport for reduced-SWIRL verifier contexts.

use std::sync::Arc;

use openvm_cpu_backend::CpuBackend;
use openvm_cuda_backend::{GpuBackend, GpuDevice};
use openvm_stark_backend::{
    p3_matrix::Matrix,
    prover::{AirProvingContext, CommittedTraceData, DeviceDataTransporter, TraceCommitter},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::Digest;

use super::reduced_swirl_error::ReducedSwirlWrapperSystemError;
use crate::SC;

/// Upload indexed verifier contexts while deduplicating cached traces with
/// identical commitments and geometry.
pub fn transport_reduced_swirl_contexts_to_cuda(
    device: &GpuDevice,
    contexts: Vec<(usize, AirProvingContext<CpuBackend<SC>>)>,
) -> Result<Vec<(usize, AirProvingContext<GpuBackend>)>, ReducedSwirlWrapperSystemError> {
    struct CachedUpload {
        commitment: Digest,
        width: usize,
        height: usize,
        committed: CommittedTraceData<GpuBackend>,
    }

    let mut cached_uploads = Vec::<CachedUpload>::new();
    let mut component_contexts = Vec::with_capacity(contexts.len());
    for (air, context) in contexts {
        let AirProvingContext {
            cached_mains,
            common_main,
            public_values,
        } = context;
        let common_main =
            <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::
                transport_row_major_matrix_to_device(device, &common_main);
        let mut device_cached = Vec::with_capacity(cached_mains.len());
        for cached in cached_mains {
            let width = Matrix::width(&cached.trace);
            let height = Matrix::height(&cached.trace);
            if let Some(existing) = cached_uploads.iter().find(|existing| {
                existing.commitment == cached.commitment
                    && existing.width == width
                    && existing.height == height
            }) {
                device_cached.push(existing.committed.clone());
                continue;
            }
            if cached_uploads
                .iter()
                .any(|existing| existing.commitment == cached.commitment)
            {
                return Err(ReducedSwirlWrapperSystemError::Context(
                    "cached commitment reused with different geometry",
                ));
            }
            let trace =
                <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::
                    transport_row_major_matrix_to_device(device, &cached.trace);
            let (commitment, data) =
                <GpuDevice as TraceCommitter<GpuBackend>>::commit(device, &[&trace]).map_err(
                    |error| {
                        ReducedSwirlWrapperSystemError::Prover(format!(
                            "CUDA cached-main commitment: {error}"
                        ))
                    },
                )?;
            if commitment != cached.commitment {
                return Err(ReducedSwirlWrapperSystemError::Context(
                    "CPU/CUDA cached commitment mismatch",
                ));
            }
            let committed = CommittedTraceData {
                commitment,
                trace,
                data: Arc::new(data),
            };
            cached_uploads.push(CachedUpload {
                commitment,
                width,
                height,
                committed: committed.clone(),
            });
            device_cached.push(committed);
        }
        component_contexts.push((
            air,
            AirProvingContext::new(device_cached, common_main, public_values),
        ));
    }
    Ok(component_contexts)
}

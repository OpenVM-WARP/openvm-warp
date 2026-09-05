use core::borrow::BorrowMut;
use std::sync::Arc;

use itertools::Itertools;
use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip, POSEIDON2_WIDTH};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, p3_maybe_rayon::prelude::*, proof::Proof,
    prover::AirProvingContext, transcript::TranscriptLog, AirRef, StarkProtocolConfig,
    SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{poseidon2_perm, BabyBearPoseidon2Config, F};
use p3_air::BaseAir;
use p3_baby_bear::Poseidon2BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::Permutation;
use tracing::trace_span;

use crate::{
    bus::CertifiedTranscriptCheckpointBus,
    system::{AirModule, BusInventory, GlobalCtxCpu, Preflight, TraceGenModule},
    transcript::{
        merkle_verify::{MerkleVerifyAir, MerkleVerifyCols},
        poseidon2::{Poseidon2Air, Poseidon2Cols, Poseidon2MultiBusAir, CHUNK},
        transcript::{
            TranscriptAir, TranscriptCheckpointCols, TranscriptCols, TranscriptResumeCols,
        },
    },
};

pub type Poseidon2MultibusInputs = Vec<(Vec<[F; POSEIDON2_WIDTH]>, Vec<[F; POSEIDON2_WIDTH]>)>;

/// One logically independent pair of Poseidon lookup buses.
///
/// A multi-bus physical table must retain one multiplicity pair per owner;
/// flattening two owners would merge distinct lookup namespaces.
#[derive(Clone, Copy, Debug)]
pub struct Poseidon2BusOwner {
    pub permute_bus: crate::bus::Poseidon2PermuteBus,
    pub compress_bus: crate::bus::Poseidon2CompressBus,
}

#[cfg(feature = "cuda")]
mod cuda_abi;
pub mod merkle_verify;
pub mod poseidon2;
#[allow(clippy::module_inception)]
pub mod transcript;

/// Number of intermediate S-box registers used by this transcript's Poseidon
/// table. Ordinary recursive circuits retain the degree-three default. A
/// setup whose global degree bound is at least seven may select zero registers
/// to trade a degree-seven constraint for roughly half the witness columns.
pub struct TranscriptModule<const SBOX_REGISTERS: usize = 1> {
    pub bus_inventory: BusInventory,
    params: SystemParams,
    final_state_bus_enabled: bool,
    end_index_bus_enabled: bool,
    /// Set when this circuit's transcripts continue an earlier one.
    resume_state_bus_enabled: bool,
    checkpoint_state_bus: Option<CertifiedTranscriptCheckpointBus>,
    merkle_verify_enabled: bool,

    sub_chip: Poseidon2SubChip<F, SBOX_REGISTERS>,
    perm: Poseidon2BabyBear<POSEIDON2_WIDTH>,
}

impl<const SBOX_REGISTERS: usize> TranscriptModule<SBOX_REGISTERS> {
    pub fn new(
        bus_inventory: BusInventory,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
    ) -> Self {
        Self::new_with_checkpoint_bus(
            bus_inventory,
            params,
            final_state_bus_enabled,
            resume_state_bus_enabled,
            None,
        )
    }

    pub fn new_with_checkpoint_bus(
        bus_inventory: BusInventory,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
        checkpoint_state_bus: Option<CertifiedTranscriptCheckpointBus>,
    ) -> Self {
        assert!(
            matches!(SBOX_REGISTERS, 0 | 1),
            "BabyBear degree-seven Poseidon supports zero or one S-box register"
        );
        let sub_chip =
            Poseidon2SubChip::<F, SBOX_REGISTERS>::new(Poseidon2Config::default().constants);
        Self {
            bus_inventory,
            params,
            final_state_bus_enabled,
            end_index_bus_enabled: false,
            resume_state_bus_enabled,
            checkpoint_state_bus,
            merkle_verify_enabled: true,
            sub_chip,
            perm: poseidon2_perm().clone(),
        }
    }

    /// Enable row-aligned transcript checkpoints for an enclosing protocol.
    ///
    /// This only changes the fixed Transcript AIR/trace layout. The caller is
    /// responsible for allocating the bus after every bus owned by this
    /// module, so enabling it cannot renumber the ordinary recursive verifier.
    pub fn set_checkpoint_state_bus(
        &mut self,
        checkpoint_state_bus: CertifiedTranscriptCheckpointBus,
    ) {
        self.checkpoint_state_bus = Some(checkpoint_state_bus);
    }

    /// Export the exact transcript operation count at the final row.
    ///
    /// This is independent of resume-state mode: a pre-stacking terminal uses
    /// it only as length binding for the public transcript checkpoint.
    pub fn enable_end_index_bus(&mut self) {
        self.end_index_bus_enabled = true;
    }

    /// Disable child PCS query authentication for a verifier that stops after
    /// stacking.  Commitment observations remain transcript-bound; a separate
    /// terminal PCS obligation must consume the exported checkpoint.
    pub fn disable_merkle_verify(&mut self) {
        self.merkle_verify_enabled = false;
    }

    #[must_use]
    pub fn poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        Poseidon2BusOwner {
            permute_bus: self.bus_inventory.poseidon2_permute_bus,
            compress_bus: self.bus_inventory.poseidon2_compress_bus,
        }
    }

    /// Build one physical Poseidon AIR with a distinct multiplicity pair for
    /// each owner, in exactly the supplied order.
    #[must_use]
    pub fn multi_bus_poseidon2_air_for_owners<SC: StarkProtocolConfig<F = F>>(
        &self,
        owners: &[Poseidon2BusOwner],
    ) -> AirRef<SC> {
        assert!(!owners.is_empty(), "Poseidon2 table needs a bus owner");
        Arc::new(Poseidon2MultiBusAir::<F, SBOX_REGISTERS> {
            subair: self.sub_chip.air.clone(),
            buses: owners
                .iter()
                .map(|owner| (owner.permute_bus, owner.compress_bus))
                .collect(),
        })
    }

    pub(crate) fn multi_bus_poseidon2_air<SC: StarkProtocolConfig<F = F>>(
        &self,
        owners: &[&Self],
    ) -> AirRef<SC> {
        let owners = owners
            .iter()
            .map(|owner| owner.poseidon2_bus_owner())
            .collect::<Vec<_>>();
        self.multi_bus_poseidon2_air_for_owners(&owners)
    }

    // Builds trace for transcript and merkle verify AIRs (and records poseidon2 permutations).
    // Also combines in the poseidon2 permutations from preflight (from WHIR).
    #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
    fn build_trace_artifacts(
        &self,
        preflights: &[Preflight],
        mut poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        mut poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_height: Option<usize>,
    ) -> Option<TranscriptTraceArtifacts> {
        for preflight in preflights {
            poseidon2_perm_inputs.extend_from_slice(&preflight.poseidon2_perm_inputs);
            poseidon2_compress_inputs.extend_from_slice(&preflight.poseidon2_compress_inputs);
        }
        let logs = preflights
            .iter()
            .map(|preflight| &preflight.transcript)
            .collect_vec();
        self.build_transcript_trace_artifacts(
            &logs,
            &[],
            &[],
            poseidon2_perm_inputs,
            poseidon2_compress_inputs,
            required_height,
        )
    }

    /// `resumes[i]`, when present, is the `(first operation index, sponge state)`
    /// that log `i` continues from instead of `(0, zero sponge)`. Supplying it
    /// lets a recursive stage carry only its own transcript span; replaying every
    /// earlier stage's to re-derive the same state is what made the history
    /// circuit's hashing quadratic in the stage count.
    pub(crate) fn build_transcript_trace_artifacts(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        checkpoints: &[Option<[usize; 2]>],
        poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_height: Option<usize>,
    ) -> Option<TranscriptTraceArtifacts> {
        let checkpoints = checkpoints
            .iter()
            .map(|targets| targets.map(|[left, right]| [Some(left), Some(right)]))
            .collect::<Vec<_>>();
        self.build_transcript_trace_artifacts_with_optional_checkpoints(
            logs,
            resumes,
            &checkpoints,
            poseidon2_perm_inputs,
            poseidon2_compress_inputs,
            required_height,
        )
    }

    /// Internal variant allowing one protocol to certify only one checkpoint
    /// kind. An absent kind emits no lookup instead of an unmatched synthetic
    /// checkpoint message.
    pub(crate) fn build_transcript_trace_artifacts_with_optional_checkpoints(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        checkpoints: &[Option<[Option<usize>; 2]>],
        mut poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_height: Option<usize>,
    ) -> Option<TranscriptTraceArtifacts> {
        if !resumes.is_empty() && resumes.len() != logs.len() {
            return None;
        }
        if self.checkpoint_state_bus.is_some() {
            if checkpoints.len() != logs.len() {
                return None;
            }
        } else if checkpoints
            .iter()
            .flatten()
            .any(|targets| targets.iter().any(Option::is_some))
        {
            return None;
        }
        // Width must follow the same flag the AIR uses, not the contents of
        // `resumes`. Deriving it from `resumes.iter().any(..)` lets a module
        // built with the bus enabled but handed `&[]` or all-`None` report one
        // width from the AIR and generate another in the trace.
        let resume_width = if self.resume_state_bus_enabled {
            TranscriptResumeCols::<F>::width()
        } else {
            0
        };
        let checkpoint_width = if self.checkpoint_state_bus.is_some() {
            TranscriptCheckpointCols::<F>::width()
        } else {
            0
        };
        if !self.resume_state_bus_enabled && resumes.iter().any(Option::is_some) {
            return None;
        }
        let base_width = TranscriptCols::<F>::width();
        let transcript_width = base_width + resume_width + checkpoint_width;
        let mut valid_rows = Vec::with_capacity(logs.len());

        let mut transcript_valid_rows = 0;
        // First pass, calculate number of rows for transcript
        for log in logs {
            let mut cur_is_sample = false; // should always start with observe?
            let mut count = 0;
            let mut num_valid_rows: usize = 0;
            for op_is_sample in log.samples() {
                if *op_is_sample {
                    // sample
                    if !cur_is_sample {
                        // observe -> sample, need a new row and permute
                        num_valid_rows += 1;
                        cur_is_sample = true;
                        count = 1;
                    } else {
                        if count == CHUNK {
                            num_valid_rows += 1;
                            count = 0;
                        }
                        count += 1;
                    }
                } else {
                    // observe
                    if cur_is_sample {
                        // sample -> observe, no need to permute, but still need a new row
                        num_valid_rows += 1;
                        cur_is_sample = false;
                        count = 1;
                    } else {
                        if count == CHUNK {
                            num_valid_rows += 1;
                            count = 0;
                        }
                        count += 1;
                    }
                }
            }
            if count > 0 {
                num_valid_rows += 1;
            }
            valid_rows.push(num_valid_rows);
            transcript_valid_rows += num_valid_rows;
        }
        let transcript_num_rows = if let Some(height) = required_height {
            if height < transcript_valid_rows {
                return None;
            }
            height
        } else {
            transcript_valid_rows.next_power_of_two()
        };
        let mut transcript_trace = vec![F::ZERO; transcript_num_rows * transcript_width];

        let mut skip = 0;
        // Second pass, fill in the transcript trace.
        for (pidx, log) in logs.iter().enumerate() {
            let resume = resumes.get(pidx).copied().flatten();
            let (base_tidx, resume_state) = resume.unwrap_or((0, [F::ZERO; POSEIDON2_WIDTH]));
            let mut tidx = 0;
            let mut prev_poseidon_state = resume_state;
            let checkpoint_targets = checkpoints.get(pidx).copied().flatten();
            let mut checkpoint_found = [false; 2];
            let off = skip * transcript_width;
            let end = off + valid_rows[pidx] * transcript_width;
            for (i, row) in transcript_trace[off..end]
                .chunks_exact_mut(transcript_width)
                .enumerate()
            {
                let (base_row, trailing) = row.split_at_mut(base_width);
                let (resume_row, checkpoint_row) = trailing.split_at_mut(resume_width);
                if resume_width != 0 {
                    let resume_cols: &mut TranscriptResumeCols<F> = resume_row.borrow_mut();
                    resume_cols.state = resume_state;
                }
                let cols: &mut TranscriptCols<F> = base_row.borrow_mut();
                cols.proof_idx = F::from_usize(pidx);
                if i == 0 {
                    cols.is_proof_start = F::ONE;
                }
                let is_sample = log.samples()[tidx];

                cols.is_sample = F::from_bool(is_sample);
                cols.tidx = F::from_usize(base_tidx + tidx);
                cols.mask[0] = F::from_bool(true);

                cols.prev_state = prev_poseidon_state;

                if is_sample {
                    debug_assert_eq!(
                        cols.prev_state[CHUNK - 1],
                        log.values()[tidx],
                        "sample value mismatch",
                    );
                } else {
                    cols.prev_state[0] = log.values()[tidx];
                }

                tidx += 1;
                let mut idx: usize = 1;

                let mut permuted = false;
                loop {
                    if tidx >= log.len() {
                        // at the end, no permutation needed
                        break;
                    }

                    if log.samples()[tidx] != is_sample {
                        // encounter a different type of operation. Permute if it's going to sample
                        permuted = log.samples()[tidx];
                        break;
                    }

                    cols.mask[idx] = F::from_bool(true);
                    if is_sample {
                        debug_assert_eq!(
                            cols.prev_state[CHUNK - 1 - idx],
                            log.values()[tidx],
                            "sample value mismatch",
                        );
                    } else {
                        cols.prev_state[idx] = log.values()[tidx];
                    }

                    tidx += 1;
                    idx += 1;
                    if idx == CHUNK {
                        // If it's sample -> observe, we don't need to permute. otherwise permute
                        permuted = tidx < log.len() && (!is_sample || log.samples()[tidx]);
                        break;
                    }
                }

                // Length binding: an observe row adds its operation count to
                // the first capacity lane before the row's permutation,
                // mirroring `DuplexSponge::absorb`'s per-absorb increment.
                if !is_sample {
                    cols.prev_state[CHUNK] += F::from_usize(idx);
                }
                prev_poseidon_state = cols.prev_state;
                if permuted {
                    self.perm.permute_mut(&mut prev_poseidon_state);
                    poseidon2_perm_inputs.push(cols.prev_state);
                }
                cols.post_state = prev_poseidon_state;
                if let Some(targets) = checkpoint_targets {
                    let checkpoint: &mut TranscriptCheckpointCols<F> = checkpoint_row.borrow_mut();
                    let absolute_end = base_tidx + tidx;
                    for (kind, target) in targets.into_iter().enumerate() {
                        let Some(target) = target else {
                            continue;
                        };
                        if absolute_end == target {
                            if !is_sample || checkpoint_found[kind] {
                                return None;
                            }
                            checkpoint.selected[kind] = F::ONE;
                            checkpoint_found[kind] = true;
                        }
                    }
                }
            }
            if let Some(targets) = checkpoint_targets {
                for kind in 0..2 {
                    if checkpoint_found[kind] != targets[kind].is_some() {
                        return None;
                    }
                }
            }
            skip += valid_rows[pidx];
            assert_eq!(tidx, log.len());
        }

        Some(TranscriptTraceArtifacts {
            transcript_trace: RowMajorMatrix::new(transcript_trace, transcript_width),
            poseidon2_perm_inputs,
            poseidon2_compress_inputs,
        })
    }

    /// Build the Poseidon2 trace directly into device memory.
    ///
    /// Deduplication stays on the host: it is a hash over the request states,
    /// small next to the permutation trace itself, which is what this avoids
    /// materialising and then copying. The reduced-SWIRL wrapper builds this
    /// trace per transition and it is 67% of the cells that transition
    /// hands to the transporter, so building it where it is consumed removes
    /// both the host tracegen and the upload.
    #[cfg(feature = "cuda")]
    pub fn build_poseidon2_trace_gpu(
        &self,
        poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_height: Option<usize>,
        device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Option<openvm_cuda_backend::base::DeviceMatrix<F>> {
        use openvm_cuda_backend::base::DeviceMatrix;
        use openvm_cuda_common::copy::MemCopyH2D;

        let _dedup_span = tracing::info_span!("poseidon2_dedup_gpu").entered();
        let (states, counts) =
            Self::dedup_poseidon_inputs(poseidon2_perm_inputs, poseidon2_compress_inputs);
        drop(_dedup_span);

        let num_records = states.len();
        let num_rows = match required_height {
            Some(height) if height == 0 || num_records > height => return None,
            Some(height) => height,
            None if num_records == 0 => 1,
            None => num_records.next_power_of_two(),
        };
        let width = Poseidon2Cols::<F, SBOX_REGISTERS>::width();
        let trace = DeviceMatrix::<F>::with_capacity_on(num_rows, width, device_ctx);
        if num_records == 0 {
            return Some(trace);
        }

        let flat: Vec<F> = states.into_iter().flatten().collect();
        let d_records = flat.to_device_on(device_ctx).ok()?;
        let d_counts = counts.to_device_on(device_ctx).ok()?;
        // SAFETY: `d_records` holds `num_records * POSEIDON2_WIDTH` elements and
        // `d_counts` one entry per record, which is what the kernel indexes.
        unsafe {
            cuda_abi::poseidon2_tracegen(
                trace.buffer(),
                num_rows,
                width,
                &d_records,
                &d_counts,
                num_records,
                SBOX_REGISTERS,
                device_ctx.stream.as_raw(),
            )
            .ok()?;
        }
        Some(trace)
    }

    /// Build one logical Poseidon2 lookup relation backed by one physical
    /// trace. Requests are deduplicated across owners, while a separate
    /// multiplicity pair per owner keeps the lookup relations disjoint.
    #[cfg(feature = "cuda")]
    pub(crate) fn build_poseidon2_multibus_traces_gpu(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
        device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Option<Vec<openvm_cuda_backend::base::DeviceMatrix<F>>> {
        use openvm_cuda_backend::base::DeviceMatrix;
        use openvm_cuda_common::copy::MemCopyH2D;

        let group_count = grouped_inputs.len();
        let _dedup_span = tracing::info_span!("poseidon2_multibus_dedup_gpu").entered();
        let (states, counts) = Self::dedup_poseidon_multibus_inputs(grouped_inputs)?;
        drop(_dedup_span);
        let inner_width = self.sub_chip.air.width();
        let width = inner_width + 2 * group_count;
        let num_records = states.len();
        let num_rows = num_records.max(1).next_power_of_two();
        tracing::info!(
            unique_requests = num_records,
            group_count,
            trace_height = num_rows,
            "native WARP multi-bus Poseidon2 layout"
        );
        let trace = DeviceMatrix::<F>::with_capacity_on(num_rows, width, device_ctx);
        let flat_records = states.iter().flatten().copied().collect_vec();
        let flat_counts = counts
            .iter()
            .flat_map(|by_group| {
                by_group
                    .iter()
                    .flat_map(|count| [count.perm, count.compress])
            })
            .collect_vec();
        let d_records = flat_records.to_device_on(device_ctx).ok()?;
        let d_counts = flat_counts.to_device_on(device_ctx).ok()?;
        // SAFETY: records contain one width-16 state and counts contain
        // `2 * group_count` u32 multiplicities per valid row.
        unsafe {
            cuda_abi::poseidon2_multibus_tracegen(
                trace.buffer(),
                num_rows,
                width,
                &d_records,
                &d_counts,
                num_records,
                group_count,
                SBOX_REGISTERS,
                device_ctx.stream.as_raw(),
            )
            .ok()?;
        }
        Some(vec![trace])
    }

    fn dedup_poseidon_inputs(
        poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> (Vec<[F; POSEIDON2_WIDTH]>, Vec<Poseidon2Count>) {
        let keyed_perm_states = poseidon2_perm_inputs
            .into_iter()
            .map(|state| (state.map(|x| x.as_canonical_u32()), state, true));
        let keyed_compress_states = poseidon2_compress_inputs
            .into_iter()
            .map(|state| (state.map(|x| x.as_canonical_u32()), state, false));
        let mut keyed_states = keyed_perm_states
            .into_iter()
            .chain(keyed_compress_states)
            .collect_vec();
        // Parallel because this sorts one entry per Poseidon2 request, and a
        // wide reduced-SWIRL transition makes ~417k of them with 64-byte keys.
        // Unstable ordering is still fine: two entries share a key only when
        // their canonical limbs agree, so they carry the same state, and the
        // dedup below only counts them.
        #[cfg(feature = "parallel")]
        keyed_states.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        #[cfg(not(feature = "parallel"))]
        keyed_states.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut deduped = Vec::new();
        let mut counts: Vec<Poseidon2Count> = Vec::new();
        let mut last_key: Option<[u32; POSEIDON2_WIDTH]> = None;

        for (key, state, is_perm) in keyed_states {
            if last_key == Some(key) {
                if is_perm {
                    counts.last_mut().unwrap().perm += 1;
                } else {
                    counts.last_mut().unwrap().compress += 1;
                }
            } else {
                deduped.push(state);
                counts.push(if is_perm {
                    Poseidon2Count {
                        perm: 1,
                        compress: 0,
                    }
                } else {
                    Poseidon2Count {
                        perm: 0,
                        compress: 1,
                    }
                });
                last_key = Some(key);
            }
        }
        (deduped, counts)
    }

    fn dedup_poseidon_multibus_inputs(
        grouped_inputs: Poseidon2MultibusInputs,
    ) -> Option<(Vec<[F; POSEIDON2_WIDTH]>, Vec<Vec<Poseidon2Count>>)> {
        let group_count = grouped_inputs.len();
        if group_count == 0 {
            return None;
        }
        let mut keyed_states = grouped_inputs
            .into_iter()
            .enumerate()
            .flat_map(|(group, (permutations, compressions))| {
                permutations
                    .into_iter()
                    .map(move |state| (state.map(|x| x.as_canonical_u32()), state, group, true))
                    .chain(compressions.into_iter().map(move |state| {
                        (state.map(|x| x.as_canonical_u32()), state, group, false)
                    }))
            })
            .collect_vec();
        #[cfg(feature = "parallel")]
        keyed_states.par_sort_unstable_by(|left, right| left.0.cmp(&right.0));
        #[cfg(not(feature = "parallel"))]
        keyed_states.sort_unstable_by(|left, right| left.0.cmp(&right.0));

        let mut deduped = Vec::new();
        let mut counts: Vec<Vec<Poseidon2Count>> = Vec::new();
        let mut last_key: Option<[u32; POSEIDON2_WIDTH]> = None;
        for (key, state, group, is_perm) in keyed_states {
            if last_key != Some(key) {
                deduped.push(state);
                counts.push(vec![Poseidon2Count::default(); group_count]);
                last_key = Some(key);
            }
            let count = counts.last_mut()?.get_mut(group)?;
            if is_perm {
                count.perm = count.perm.checked_add(1)?;
            } else {
                count.compress = count.compress.checked_add(1)?;
            }
        }
        Some((deduped, counts))
    }

    pub(crate) fn build_poseidon2_trace(
        &self,
        poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let _dedup_span = tracing::info_span!("poseidon2_dedup").entered();
        let (mut poseidon_states, poseidon_counts) =
            Self::dedup_poseidon_inputs(poseidon2_perm_inputs, poseidon2_compress_inputs);
        drop(_dedup_span);
        let valid_rows = poseidon_states.len();
        let num_rows = if let Some(height) = required_height {
            if height == 0 || valid_rows > height {
                return None;
            }
            height
        } else if valid_rows == 0 {
            1
        } else {
            valid_rows.next_power_of_two()
        };
        poseidon_states.resize(num_rows, [F::ZERO; POSEIDON2_WIDTH]);
        let inner_width = self.sub_chip.air.width();
        let width = Poseidon2Cols::<F, SBOX_REGISTERS>::width();
        let _inner_span = tracing::info_span!("poseidon2_inner_tracegen").entered();
        let inner_trace = self.sub_chip.generate_trace(poseidon_states);
        drop(_inner_span);
        let _widen_span = tracing::info_span!("poseidon2_widen").entered();
        let mut trace = F::zero_vec(num_rows * width);
        trace
            .par_chunks_mut(width)
            .zip(inner_trace.values.par_chunks(inner_width))
            .enumerate()
            .for_each(|(index, (row, inner_row))| {
                row[..inner_width].copy_from_slice(inner_row);
                let cols: &mut Poseidon2Cols<F, SBOX_REGISTERS> = row.borrow_mut();
                let count = poseidon_counts.get(index).copied().unwrap_or_default();
                cols.permute_mult = F::from_u32(count.perm);
                cols.compress_mult = F::from_u32(count.compress);
            });
        drop(_widen_span);
        Some(RowMajorMatrix::new(trace, width))
    }

    pub fn build_poseidon2_multibus_traces(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
    ) -> Option<Vec<RowMajorMatrix<F>>> {
        self.build_poseidon2_multibus_sharded_traces(
            grouped_inputs,
            1,
            1usize << (usize::BITS as usize - 1),
        )
    }

    /// Build a setup-fixed number of physical tables for one logical
    /// multi-bus Poseidon relation.
    ///
    /// Requests are deduplicated globally first, then the ordered unique
    /// states are partitioned into tables of at most `max_rows`. Every table
    /// carries the same owner/bus columns, so splitting only changes the
    /// physical trace layout: all lookup multiplicities still contribute to
    /// the exact same authenticated buses. Empty setup-reserved shards are
    /// represented by one inactive row.
    pub fn build_poseidon2_multibus_sharded_traces(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
        shard_count: usize,
        max_rows: usize,
    ) -> Option<Vec<RowMajorMatrix<F>>> {
        if shard_count == 0 || max_rows == 0 || !max_rows.is_power_of_two() {
            return None;
        }
        let group_count = grouped_inputs.len();
        let (states, counts) = Self::dedup_poseidon_multibus_inputs(grouped_inputs)?;
        if states.len() > shard_count.checked_mul(max_rows)? {
            return None;
        }
        let inner_width = self.sub_chip.air.width();
        let width = inner_width + 2 * group_count;
        (0..shard_count)
            .map(|shard| {
                let start = shard.checked_mul(max_rows)?;
                let end = states.len().min(start.saturating_add(max_rows));
                let valid_rows = end.saturating_sub(start);
                let num_rows = valid_rows.max(1).next_power_of_two();
                let mut poseidon_states = if start < states.len() {
                    states[start..end].to_vec()
                } else {
                    Vec::new()
                };
                poseidon_states.resize(num_rows, [F::ZERO; POSEIDON2_WIDTH]);
                let inner_trace = self.sub_chip.generate_trace(poseidon_states);
                let mut trace = F::zero_vec(num_rows * width);
                trace
                    .par_chunks_mut(width)
                    .zip(inner_trace.values.par_chunks(inner_width))
                    .enumerate()
                    .for_each(|(local_index, (row, inner_row))| {
                        row[..inner_width].copy_from_slice(inner_row);
                        if local_index < valid_rows {
                            let by_group = &counts[start + local_index];
                            for (group, count) in by_group.iter().enumerate() {
                                row[inner_width + 2 * group] = F::from_u32(count.perm);
                                row[inner_width + 2 * group + 1] = F::from_u32(count.compress);
                            }
                        }
                    });
                Some(RowMajorMatrix::new(trace, width))
            })
            .collect()
    }
}

impl<const SBOX_REGISTERS: usize> AirModule for TranscriptModule<SBOX_REGISTERS> {
    fn num_airs(&self) -> usize {
        3
    }

    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let transcript_air = TranscriptAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            poseidon2_permute_bus: self.bus_inventory.poseidon2_permute_bus,
            final_state_bus: self
                .final_state_bus_enabled
                .then_some(self.bus_inventory.final_state_bus),
            resume_state_bus: self
                .resume_state_bus_enabled
                .then_some(self.bus_inventory.resume_state_bus),
            end_index_bus: (self.resume_state_bus_enabled || self.end_index_bus_enabled)
                .then_some(self.bus_inventory.transcript_end_index_bus),
            checkpoint_state_bus: self.checkpoint_state_bus,
        };
        let poseidon2_air = Poseidon2Air::<F, SBOX_REGISTERS> {
            subair: self.sub_chip.air.clone(),
            poseidon2_permute_bus: self.bus_inventory.poseidon2_permute_bus,
            poseidon2_compress_bus: self.bus_inventory.poseidon2_compress_bus,
        };
        let merkle_verify_air = MerkleVerifyAir {
            poseidon2_compress_bus: self.bus_inventory.poseidon2_compress_bus,
            merkle_verify_bus: self.bus_inventory.merkle_verify_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
            right_shift_bus: self.bus_inventory.right_shift_bus,
            k: self.params.k_whir(),
        };
        vec![
            Arc::new(transcript_air),
            Arc::new(poseidon2_air),
            Arc::new(merkle_verify_air),
        ]
    }
}

pub(crate) struct TranscriptTraceArtifacts {
    pub(crate) transcript_trace: RowMajorMatrix<F>,
    pub(crate) poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub(crate) poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(super) struct Poseidon2Count {
    pub perm: u32,
    pub compress: u32,
}

impl<SC: StarkProtocolConfig<F = F>, const SBOX_REGISTERS: usize>
    TraceGenModule<GlobalCtxCpu, CpuBackend<SC>> for TranscriptModule<SBOX_REGISTERS>
{
    // External Poseidon2 compress inputs
    type ModuleSpecificCtx<'a> = (&'a Vec<[F; POSEIDON2_WIDTH]>, &'a Vec<[F; POSEIDON2_WIDTH]>);

    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        ctx: &Self::ModuleSpecificCtx<'_>,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<SC>>>> {
        let external_poseidon2_permute_inputs = ctx.0;
        let external_poseidon2_compress_inputs = ctx.1;
        let (required_transcript, required_poseidon2, required_merkle_verify) =
            if let Some(heights) = required_heights {
                if heights.len() != 3 {
                    return None;
                }
                (Some(heights[0]), Some(heights[1]), Some(heights[2]))
            } else {
                (None, None, None)
            };

        let (merkle_verify_trace_vec, poseidon2_compress_inputs) = if self.merkle_verify_enabled {
            tracing::info_span!("wrapper.generate_trace", air = "MerkleVerify").in_scope(|| {
                merkle_verify::generate_trace(
                    child_vk,
                    proofs,
                    preflights,
                    &self.params,
                    required_merkle_verify,
                )
            })?
        } else {
            let height = required_merkle_verify.unwrap_or(1);
            if height == 0 || !height.is_power_of_two() {
                return None;
            }
            (
                F::zero_vec(height * MerkleVerifyCols::<F>::width()),
                Vec::new(),
            )
        };
        let merkle_verify_trace =
            RowMajorMatrix::new(merkle_verify_trace_vec, MerkleVerifyCols::<F>::width());
        let TranscriptTraceArtifacts {
            transcript_trace,
            mut poseidon2_perm_inputs,
            mut poseidon2_compress_inputs,
        } = tracing::trace_span!("wrapper.generate_trace", air = "Transcript").in_scope(|| {
            self.build_trace_artifacts(
                preflights,
                vec![],
                poseidon2_compress_inputs,
                required_transcript,
            )
        })?;
        poseidon2_perm_inputs.extend_from_slice(external_poseidon2_permute_inputs);
        poseidon2_compress_inputs.extend_from_slice(external_poseidon2_compress_inputs);

        let poseidon2_trace =
            trace_span!("wrapper.generate_trace", air = "Poseidon2").in_scope(|| {
                self.build_poseidon2_trace(
                    poseidon2_perm_inputs,
                    poseidon2_compress_inputs,
                    required_poseidon2,
                )
            })?;

        // Finally, make the RawInput structs
        Some(
            [transcript_trace, poseidon2_trace, merkle_verify_trace]
                .map(AirProvingContext::simple_no_pis)
                .into_iter()
                .collect(),
        )
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda_tracegen {
    use itertools::Itertools;
    use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::GpuDeviceCtx,
    };
    use openvm_stark_backend::prover::MatrixDimensions;

    use super::*;
    use crate::{
        cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu, GlobalCtxGpu},
        transcript::{
            cuda_abi,
            merkle_verify::{self, cuda::MerkleVerifyBlob},
            transcript::cuda::TranscriptAirBlob,
        },
    };

    pub(crate) struct TranscriptBlob {
        pub merkle_verify_blob: MerkleVerifyBlob,
        pub transcript_air_blob: TranscriptAirBlob,

        // Because we currently can only copy to the beginning of a DeviceBuffer, the layout is
        // expected to be in this order:
        // - Preflight permutations
        // - Preflight compressions
        // - Merkle verify compressions
        // - Transcript permutations
        pub poseidon2_buffer: DeviceBuffer<F>,
        pub num_prefix_perms: usize,
        pub num_suffix_perms: usize,
        pub num_compress_inputs: usize,
    }

    impl TranscriptBlob {
        #[tracing::instrument(name = "generate_blob", skip_all)]
        pub fn new(
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            external_poseidon2_inputs: &(
                &Vec<[F; POSEIDON2_WIDTH]>,
                &Vec<[F; POSEIDON2_WIDTH]>,
                &GpuDeviceCtx,
            ),
        ) -> Self {
            Self::new_inner(
                child_vk,
                proofs,
                preflights,
                external_poseidon2_inputs,
                true,
            )
            .expect("Merkle inputs must match verifier preflight")
        }

        /// Builds the transcript-side CUDA data for a verifier that stops
        /// before the terminal PCS opening. The preflight transcript and all
        /// non-Merkle Poseidon operations are retained, but no WHIR path is
        /// reconstructed or hashed.
        pub fn new_without_merkle(
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            external_poseidon2_inputs: &(
                &Vec<[F; POSEIDON2_WIDTH]>,
                &Vec<[F; POSEIDON2_WIDTH]>,
                &GpuDeviceCtx,
            ),
        ) -> Option<Self> {
            Self::new_inner(
                child_vk,
                proofs,
                preflights,
                external_poseidon2_inputs,
                false,
            )
        }

        fn new_inner(
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            external_poseidon2_inputs: &(
                &Vec<[F; POSEIDON2_WIDTH]>,
                &Vec<[F; POSEIDON2_WIDTH]>,
                &GpuDeviceCtx,
            ),
            merkle_verify_enabled: bool,
        ) -> Option<Self> {
            let external_poseidon2_permute_inputs = external_poseidon2_inputs.0;
            let external_poseidon2_compress_inputs = external_poseidon2_inputs.1;
            let device_ctx = external_poseidon2_inputs.2;
            let poseidon2_perm_inputs = preflights
                .iter()
                .flat_map(|preflight| preflight.cpu.poseidon2_perm_inputs.clone())
                .chain(external_poseidon2_permute_inputs.iter().copied())
                .collect_vec();
            let poseidon2_compress_inputs = preflights
                .iter()
                .flat_map(|preflight| preflight.cpu.poseidon2_compress_inputs.clone())
                .chain(external_poseidon2_compress_inputs.iter().copied())
                .collect_vec();
            let num_prefix_perms = poseidon2_perm_inputs.len();
            let mut num_compress_inputs = poseidon2_compress_inputs.len();

            let merkle_verify_blob = if merkle_verify_enabled {
                MerkleVerifyBlob::new(
                    child_vk,
                    proofs,
                    preflights,
                    num_prefix_perms + num_compress_inputs,
                )
            } else {
                if proofs.len() != preflights.len() {
                    return None;
                }
                MerkleVerifyBlob::empty(num_prefix_perms + num_compress_inputs, proofs.len())
            };
            num_compress_inputs += merkle_verify_blob.total_rows;

            let transcript_air_blob =
                TranscriptAirBlob::new(preflights, (num_prefix_perms + num_compress_inputs) as u32);
            let num_suffix_perms = transcript_air_blob.num_poseidon2_perms;

            let mut poseidon2_buffer = DeviceBuffer::with_capacity_on(
                (num_prefix_perms + num_compress_inputs + num_suffix_perms) * POSEIDON2_WIDTH,
                device_ctx,
            );
            poseidon2_perm_inputs
                .into_iter()
                .flatten()
                .chain(poseidon2_compress_inputs.into_iter().flatten())
                .collect_vec()
                .copy_to_on(&mut poseidon2_buffer, device_ctx)
                .unwrap();

            Some(Self {
                merkle_verify_blob,
                transcript_air_blob,
                poseidon2_buffer,
                num_prefix_perms,
                num_suffix_perms,
                num_compress_inputs,
            })
        }
    }

    impl<const SBOX_REGISTERS: usize> TraceGenModule<GlobalCtxGpu, GpuBackend>
        for TranscriptModule<SBOX_REGISTERS>
    {
        type ModuleSpecificCtx<'a> = (
            &'a Vec<[F; POSEIDON2_WIDTH]>,
            &'a Vec<[F; POSEIDON2_WIDTH]>,
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
            let device_ctx = ctx.2;
            let (required_transcript, required_poseidon2, required_merkle_verify) =
                if let Some(heights) = required_heights {
                    if heights.len() != 3 {
                        return None;
                    }
                    (Some(heights[0]), Some(heights[1]), Some(heights[2]))
                } else {
                    (None, None, None)
                };
            let mut blob = if self.merkle_verify_enabled {
                TranscriptBlob::new(child_vk, proofs, preflights, ctx)
            } else {
                TranscriptBlob::new_without_merkle(child_vk, proofs, preflights, ctx)?
            };

            let merkle_trace = tracing::trace_span!("wrapper.generate_trace", air = "MerkleVerify")
                .in_scope(|| {
                    merkle_verify::cuda::generate_trace(&blob, device_ctx, required_merkle_verify)
                })?;
            let transcript_trace = tracing::trace_span!(
                "wrapper.generate_trace",
                air = "Transcript"
            )
            .in_scope(|| {
                transcript::cuda::generate_trace(preflights, &blob, device_ctx, required_transcript)
            })?;
            let poseidon_trace = trace_span!("wrapper.generate_trace", air = "Poseidon2")
                .in_scope(|| {
                    trace_span!("generate_trace").in_scope(|| {
                        let poseidon2_width = Poseidon2Cols::<F, SBOX_REGISTERS>::width();
                        let total_poseidon2_inputs = blob.num_prefix_perms
                            + blob.num_compress_inputs
                            + blob.num_suffix_perms;

                        let d_records_dedup = if total_poseidon2_inputs == 0 {
                            DeviceBuffer::<F>::new()
                        } else {
                            DeviceBuffer::<F>::with_capacity_on(
                                total_poseidon2_inputs * POSEIDON2_WIDTH,
                                device_ctx,
                            )
                        };
                        let d_counts_dedup = if total_poseidon2_inputs == 0 {
                            DeviceBuffer::<Poseidon2Count>::new()
                        } else {
                            DeviceBuffer::<Poseidon2Count>::with_capacity_on(
                                total_poseidon2_inputs,
                                device_ctx,
                            )
                        };

                        let mut num_records = total_poseidon2_inputs;
                        if num_records > 0 {
                            // This buffer is needed only by CUB's sort/reduce pass. Keep it
                            // inside the synchronized dedup scope so it and the raw input
                            // oracle are released before allocating the (potentially multi-GiB)
                            // power-of-two Poseidon trace below.
                            let d_counts = DeviceBuffer::<Poseidon2Count>::with_capacity_on(
                                total_poseidon2_inputs,
                                device_ctx,
                            );
                            unsafe {
                                let d_num_records = [num_records].to_device_on(device_ctx).unwrap();
                                let mut temp_bytes = 0;
                                cuda_abi::poseidon2_deduplicate_records_get_temp_bytes(
                                    &blob.poseidon2_buffer,
                                    &d_counts,
                                    num_records,
                                    &d_num_records,
                                    &mut temp_bytes,
                                    device_ctx.stream.as_raw(),
                                )
                                .unwrap();
                                let d_temp_storage = if temp_bytes == 0 {
                                    DeviceBuffer::<u8>::new()
                                } else {
                                    DeviceBuffer::<u8>::with_capacity_on(temp_bytes, device_ctx)
                                };
                                cuda_abi::poseidon2_deduplicate_records(
                                    &blob.poseidon2_buffer,
                                    &d_counts,
                                    &d_records_dedup,
                                    &d_counts_dedup,
                                    num_records,
                                    &d_num_records,
                                    blob.num_prefix_perms,
                                    blob.num_compress_inputs,
                                    blob.num_suffix_perms,
                                    &d_temp_storage,
                                    temp_bytes,
                                    device_ctx.stream.as_raw(),
                                )
                                .unwrap();
                                num_records = *d_num_records
                                    .to_host_on(device_ctx)
                                    .unwrap()
                                    .first()
                                    .unwrap();
                            }
                        }
                        // `to_host_on` above synchronizes the caller-owned stream, so no
                        // dedup/transcript kernel can still read the raw records.  Retaining
                        // them while allocating the padded output was a pure peak-memory bug.
                        blob.poseidon2_buffer = DeviceBuffer::new();
                        let poseidon2_num_rows = if let Some(height) = required_poseidon2 {
                            if height < num_records {
                                return None;
                            }
                            height
                        } else if num_records == 0 {
                            1
                        } else {
                            num_records.next_power_of_two()
                        };
                        let poseidon_trace_gpu = DeviceMatrix::<F>::with_capacity_on(
                            poseidon2_num_rows,
                            poseidon2_width,
                            device_ctx,
                        );
                        unsafe {
                            cuda_abi::poseidon2_tracegen(
                                poseidon_trace_gpu.buffer(),
                                poseidon_trace_gpu.height(),
                                poseidon_trace_gpu.width(),
                                &d_records_dedup,
                                &d_counts_dedup,
                                num_records,
                                SBOX_REGISTERS,
                                device_ctx.stream.as_raw(),
                            )
                            .unwrap();
                        }
                        Some(poseidon_trace_gpu)
                    })
                })?;

            Some(vec![
                AirProvingContext::simple_no_pis(transcript_trace),
                AirProvingContext::simple_no_pis(poseidon_trace),
                AirProvingContext::simple_no_pis(merkle_trace),
            ])
        }
    }
}

#[cfg(test)]
mod row_count_tests {
    use core::borrow::Borrow;

    use openvm_stark_backend::{
        transcript::{TranscriptHistory, TranscriptLog},
        FiatShamirTranscript,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        default_duplex_sponge_recorder, BabyBearPoseidon2Config, F,
    };
    use p3_field::PrimeCharacteristicRing;

    type SC = BabyBearPoseidon2Config;

    /// Build a log from a script of operations: `true` samples, `false` observes.
    fn log_from(script: &[bool]) -> TranscriptLog<F, [F; 16]> {
        let mut recorder = default_duplex_sponge_recorder();
        for (index, &is_sample) in script.iter().enumerate() {
            if is_sample {
                let _ = <_ as FiatShamirTranscript<SC>>::sample(&mut recorder);
            } else {
                <_ as FiatShamirTranscript<SC>>::observe(&mut recorder, F::from_usize(index + 1));
            }
        }
        recorder.into_log()
    }

    /// The row count computed up front must match the rows the fill actually
    /// walks, or the fill indexes past the log. Closing a batch boundary with a
    /// squeeze puts lone samples into the operation stream, which is what made
    /// this matter.
    fn assert_row_count_matches_fill(script: &[bool]) {
        let log = log_from(script);
        let module = super::TranscriptModule::<1>::new(
            crate::system::BusInventory::new(&mut crate::system::BusIndexManager::new()),
            openvm_stark_sdk::config::internal_params_with_100_bits_security(),
            false,
            false,
        );
        let artifacts = module
            .build_transcript_trace_artifacts(&[&log], &[], &[], Vec::new(), Vec::new(), None)
            .expect("trace artifacts for a well-formed log");
        assert!(
            !artifacts.transcript_trace.values.is_empty(),
            "script {script:?} produced no trace"
        );
    }

    /// A transcript whose very first operation is a sample is not a shape this
    /// generator supports: a fresh sponge has to permute before it can squeeze,
    /// and the fill models a row's `prev_state` as the value read from, not the
    /// pre-permutation state. Anything that closes a rate block must therefore
    /// skip an empty transcript rather than squeeze it.
    #[test]
    fn a_log_that_starts_with_a_sample_is_not_supported() {
        let log = log_from(&[true, false, false]);
        assert_eq!(log.samples().first(), Some(&true));
    }

    #[test]
    fn a_log_that_ends_with_a_lone_sample_does_not_over_count_rows() {
        for observes in 1..=9usize {
            let mut script = vec![false; observes];
            script.push(true);
            assert_row_count_matches_fill(&script);
        }
    }

    /// Sweep the shapes a boundary squeeze actually produces: a run of
    /// observes, a run of samples that may cross the rate width, then more
    /// observes. Real transcripts sample extension elements, so a squeeze
    /// lands next to runs of four.
    #[test]
    fn boundary_squeeze_shapes_do_not_over_count_rows() {
        for leading in 0..=10usize {
            for samples in 1..=11usize {
                for trailing in 0..=10usize {
                    let mut script = vec![false; leading];
                    script.extend(vec![true; samples]);
                    script.extend(vec![false; trailing]);
                    if script.first() == Some(&true) {
                        // Covered separately; a log opening with a sample is a
                        // distinct shape from one a boundary squeeze creates.
                        continue;
                    }
                    assert_row_count_matches_fill(&script);
                }
            }
        }
    }

    #[test]
    fn samples_interleaved_with_observes_do_not_over_count_rows() {
        for observes in 1..=9usize {
            let mut script = vec![false; observes];
            script.push(true);
            script.extend(vec![false; 3]);
            script.push(true);
            assert_row_count_matches_fill(&script);
        }
    }

    #[test]
    fn a_single_optional_checkpoint_emits_no_second_kind() {
        let log = log_from(&[false, false, true, true, true, true]);
        let mut manager = crate::system::BusIndexManager::new();
        let inventory = crate::system::BusInventory::new(&mut manager);
        let checkpoint_bus =
            crate::bus::CertifiedTranscriptCheckpointBus::new(manager.new_bus_idx());
        let module = super::TranscriptModule::<1>::new_with_checkpoint_bus(
            inventory,
            openvm_stark_sdk::config::internal_params_with_100_bits_security(),
            false,
            false,
            Some(checkpoint_bus),
        );
        let artifacts = module
            .build_transcript_trace_artifacts_with_optional_checkpoints(
                &[&log],
                &[],
                &[Some([Some(log.len()), None])],
                Vec::new(),
                Vec::new(),
                None,
            )
            .expect("one checkpoint kind is a complete transcript witness");
        let base_width = super::TranscriptCols::<F>::width();
        let row_width = artifacts.transcript_trace.width;
        let mut selected = [0usize; 2];
        for row in artifacts.transcript_trace.values.chunks_exact(row_width) {
            let checkpoint: &super::TranscriptCheckpointCols<F> = row[base_width..].borrow();
            for (kind, value) in checkpoint.selected.iter().enumerate() {
                selected[kind] += usize::from(*value == F::ONE);
            }
        }
        assert_eq!(selected, [1, 0]);
    }

    #[test]
    fn physical_poseidon_shards_preserve_every_owner_multiplicity() {
        use p3_field::PrimeField32;
        use p3_matrix::Matrix;

        let mut manager = crate::system::BusIndexManager::new();
        let inventory = crate::system::BusInventory::new(&mut manager);
        let transcript = super::TranscriptModule::<1>::new(
            inventory,
            openvm_stark_backend::SystemParams::new_for_testing(4),
            false,
            false,
        );
        let a = [F::from_u32(11); super::POSEIDON2_WIDTH];
        let b = [F::from_u32(12); super::POSEIDON2_WIDTH];
        let c = [F::from_u32(13); super::POSEIDON2_WIDTH];
        let traces = transcript
            .build_poseidon2_multibus_sharded_traces(
                vec![(vec![a, a], vec![b]), (vec![a], vec![c])],
                4,
                2,
            )
            .unwrap();
        assert_eq!(traces.len(), 4);
        assert_eq!(
            traces.iter().map(Matrix::height).collect::<Vec<_>>(),
            [2, 1, 1, 1]
        );

        let width = traces[0].width();
        let multiplicity_sums = (0..4)
            .map(|column| {
                traces
                    .iter()
                    .flat_map(|trace| trace.values.chunks_exact(width))
                    .map(|row| row[width - 4 + column].as_canonical_u32())
                    .sum::<u32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(multiplicity_sums, [2, 1, 1, 1]);
    }

    #[test]
    fn physical_poseidon_shards_fail_closed_above_setup_capacity() {
        let mut manager = crate::system::BusIndexManager::new();
        let inventory = crate::system::BusInventory::new(&mut manager);
        let transcript = super::TranscriptModule::<1>::new(
            inventory,
            openvm_stark_backend::SystemParams::new_for_testing(4),
            false,
            false,
        );
        let inputs = (0..5)
            .map(|value| [F::from_u32(value + 1); super::POSEIDON2_WIDTH])
            .collect::<Vec<_>>();
        assert!(transcript
            .build_poseidon2_multibus_sharded_traces(vec![(inputs, Vec::new())], 2, 2)
            .is_none());
    }
}

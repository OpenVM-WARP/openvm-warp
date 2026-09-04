use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    time::Instant,
};

use clap::Parser;
use openvm_circuit::{
    arch::{instructions::exe::VmExe, ContinuationVmProof},
    system::memory::{merkle::public_values::UserPublicValuesProof, DIGEST_WIDTH},
};
use openvm_sdk::{
    config::{AggregationSystemParams, AppConfig},
    keygen::AppVerifyingKey,
    prover::verify_app_proof,
    DefaultStarkEngine, Sdk, StdIn, F, SC,
};
use openvm_sdk_config::{SdkVmConfig, TranspilerConfig};
use openvm_stark_backend::{
    codec::{Decode, EncodableConfig, Encode},
    proof::{Proof, PROOF_CODEC_VERSION},
    SystemParams,
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    config::{
        app_params_with_100_bits_security, internal_params_with_100_bits_security,
        leaf_params_with_100_bits_security, native_warp_app_params_with_100_bits_security,
    },
};
use openvm_transpiler::{elf::Elf, FromElf};
use openvm_verify_stark_host::{verify_vm_stark_proof_decoded, vk::VmStarkVerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

pub const DEFAULT_LOG_STACKED_HEIGHT: usize = 21;

/// Where key generation is charged relative to the measured proving window.
///
/// Both lanes generate proving keys lazily, so a lane that warms its keys
/// before the timer while the other does not is not measuring the same thing.
/// The recursive lane pays app keygen and `agg_keygen` inside `Sdk::prove`;
/// native WARP pays app keygen plus the history stage/normalization ladder.
/// The lifecycle is therefore selected once and applied to both lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BenchmarkLifecycle {
    /// Every key generation is inside `benchmark_prove_time_ms`. Setup is zero.
    #[default]
    Cold,
    /// Key generation runs before the timer and is reported as
    /// `benchmark_setup_time_ms`. Proving measures online work only.
    Warm,
}

impl BenchmarkLifecycle {
    /// Reads `OPENVM_BENCH_LIFECYCLE`, defaulting to [`Self::Cold`].
    ///
    /// Cold is the default because it is the lifecycle both lanes have always
    /// been measured under, so it stays comparable with recorded baselines.
    pub fn from_env() -> eyre::Result<Self> {
        match std::env::var("OPENVM_BENCH_LIFECYCLE") {
            Err(_) => Ok(Self::Cold),
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "" | "cold" => Ok(Self::Cold),
                "warm" => Ok(Self::Warm),
                other => {
                    eyre::bail!("OPENVM_BENCH_LIFECYCLE must be `cold` or `warm`, got `{other}`")
                }
            },
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
        }
    }

    pub const fn is_warm(self) -> bool {
        matches!(self, Self::Warm)
    }
}

/// Number of verifications timed when reporting median and p95.
///
/// Verification is milliseconds of work, so a single sample is dominated by
/// cache and scheduler noise: the same recursive proof has been observed at
/// 19.95 ms and 13.07 ms in back-to-back runs. An odd count keeps the median
/// a real sample rather than an interpolation.
const VERIFY_SAMPLES: usize = 11;

/// Reads `OPENVM_BENCH_VERIFY_SAMPLES`, defaulting to [`VERIFY_SAMPLES`].
///
/// Set it to 1 to recover single-shot behaviour for a quick run.
fn verify_sample_count() -> eyre::Result<usize> {
    match std::env::var("OPENVM_BENCH_VERIFY_SAMPLES") {
        Err(_) => Ok(VERIFY_SAMPLES),
        Ok(value) => {
            let parsed: usize = value.trim().parse().map_err(|_| {
                eyre::eyre!("OPENVM_BENCH_VERIFY_SAMPLES must be a positive integer")
            })?;
            if parsed == 0 {
                eyre::bail!("OPENVM_BENCH_VERIFY_SAMPLES must be at least 1");
            }
            Ok(parsed)
        }
    }
}

/// Times `verify` repeatedly and reports first/median/p95 in milliseconds.
///
/// The first sample is kept separately because it carries cold-cache effects
/// that the median deliberately discards; reporting both makes it visible
/// whether a lane benefits from warm caches more than the other.
fn measure_verify<V>(mut verify: V) -> eyre::Result<f64>
where
    V: FnMut() -> eyre::Result<()>,
{
    let samples = verify_sample_count()?;
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        verify()?;
        timings.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let first = timings[0];
    let mut sorted = timings;
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("verify timings are finite"));
    let median = sorted[sorted.len() / 2];
    // Nearest-rank p95, clamped to the last index for small sample counts.
    let p95 = sorted[(((sorted.len() as f64) * 0.95).ceil() as usize).saturating_sub(1)];

    // One extra pass purely for attribution. Per-phase gauges are overwritten
    // by every verification, so after a timing loop they hold the *last*
    // sample while the headline number is a median — comparing them produced
    // impossible readings such as a phase exceeding the total, or negative
    // unattributed time. Running one final pass and publishing its own total
    // alongside makes the phase gauges and `attribution_total_ms` describe the
    // same verification.
    let attribution_started = Instant::now();
    verify()?;
    let attribution_total = attribution_started.elapsed().as_secs_f64() * 1000.0;

    tracing::info!("Benchmark Verify Time (ms): {median:.3}");
    tracing::info!(
        samples = sorted.len(),
        first_ms = first,
        median_ms = median,
        p95_ms = p95,
        min_ms = sorted[0],
        max_ms = sorted[sorted.len() - 1],
        attribution_total_ms = attribution_total,
        "benchmark verify distribution"
    );
    #[cfg(feature = "metrics")]
    {
        metrics::gauge!("benchmark_verify_time_ms").set(median);
        metrics::gauge!("benchmark_verify_first_ms").set(first);
        metrics::gauge!("benchmark_verify_p95_ms").set(p95);
        metrics::gauge!("benchmark_verify_samples").set(sorted.len() as f64);
        metrics::gauge!("benchmark_verify_attribution_total_ms").set(attribution_total);
    }
    Ok(median)
}

/// Records the setup/proving split for one lane under one lifecycle.
/// Host/device traffic since process start, so the two lanes can be compared
/// on how much they move across PCIe rather than only on wall clock.
#[cfg(feature = "cuda")]
fn report_pcie_traffic() {
    let (h2d_calls, h2d_bytes, d2h_calls, d2h_bytes, d2d_calls, d2d_bytes) =
        openvm_cuda_common::copy::pcie_counters::snapshot();
    let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    tracing::info!(
        h2d_calls,
        h2d_mib = mib(h2d_bytes),
        d2h_calls,
        d2h_mib = mib(d2h_bytes),
        d2d_calls,
        d2d_mib = mib(d2d_bytes),
        "PCIe traffic"
    );
    #[cfg(feature = "metrics")]
    {
        metrics::gauge!("pcie.h2d_calls").set(h2d_calls as f64);
        metrics::gauge!("pcie.h2d_mib").set(mib(h2d_bytes));
        metrics::gauge!("pcie.d2h_calls").set(d2h_calls as f64);
        metrics::gauge!("pcie.d2h_mib").set(mib(d2h_bytes));
    }
}

#[cfg(feature = "cuda")]
fn reset_pcie_traffic() {
    openvm_cuda_common::copy::pcie_counters::reset();
}

#[cfg(not(feature = "cuda"))]
fn report_pcie_traffic() {}

#[cfg(not(feature = "cuda"))]
fn reset_pcie_traffic() {}

fn report_lifecycle(lifecycle: BenchmarkLifecycle, setup_ms: f64, prove_ms: f64) {
    report_pcie_traffic();
    tracing::info!("Benchmark Lifecycle: {}", lifecycle.as_str());
    tracing::info!("Benchmark Setup Time (ms): {setup_ms:.3}");
    tracing::info!("Benchmark Prove Time (ms): {prove_ms:.3}");
    #[cfg(feature = "metrics")]
    {
        metrics::gauge!("benchmark_setup_time_ms").set(setup_ms);
        metrics::gauge!("benchmark_prove_time_ms").set(prove_ms);
        metrics::gauge!("benchmark_lifecycle_warm").set(f64::from(u8::from(lifecycle.is_warm())));
    }
}

#[derive(Parser, Debug)]
#[command(allow_external_subcommands = true)]
pub struct BenchmarkCli {
    /// Only runs the app proof
    #[arg(long)]
    pub app_only: bool,

    /// Prove and verify the raw continuation segments, then write a versioned,
    /// self-checking capture for fixed verifier-PESAT profiling. The directory
    /// must not already exist. No recursive aggregation is run.
    #[arg(
        long,
        value_name = "DIR",
        conflicts_with_all = ["app_only", "warp", "evm"]
    )]
    pub capture_segment_proofs: Option<PathBuf>,

    /// Runs finite sparse native AIR/LogUp WARP accumulation.
    #[arg(long)]
    pub warp: bool,

    /// Segmenter's per-segment memory target in GiB. Every proving lane retains
    /// the OpenVM default when omitted.
    #[arg(long, value_name = "GIB")]
    pub segment_memory_gib: Option<usize>,

    /// Segmenter's per-segment memory target in MiB.
    #[arg(long, value_name = "MIB", conflicts_with = "segment_memory_gib")]
    pub segment_memory_mib: Option<usize>,

    /// Run full e2e proving (app → aggregation → root → halo2 wrapping)
    #[arg(long)]
    pub evm: bool,

    /// Halo2 wrapper circuit degree `k` (for e2e proving). If omitted, auto-tuned.
    #[arg(long)]
    pub halo2_wrapper_k: Option<usize>,

    /// Directory containing KZG trusted setup files (for e2e halo2 proving)
    #[arg(long)]
    pub kzg_params_dir: Option<std::path::PathBuf>,
}

impl BenchmarkCli {
    pub fn run(&self, mut vm_config: SdkVmConfig, elf: Elf, stdin: StdIn) -> eyre::Result<()> {
        let segment_memory_mib = self
            .segment_memory_mib
            .or_else(|| self.segment_memory_gib.and_then(|gib| gib.checked_shl(10)));
        if let Some(mib) = segment_memory_mib {
            let bytes = mib
                .checked_shl(20)
                .ok_or_else(|| eyre::eyre!("segment memory value is too large"))?;
            if bytes == 0 {
                eyre::bail!("segment memory must be at least 1 MiB");
            }
            vm_config.system.config.set_segmentation_max_memory(bytes);
            tracing::info!(segment_memory_mib = mib, "benchmark segment memory target");
        }
        if let Some(output_dir) = &self.capture_segment_proofs {
            run_default_segment_proof_capture(vm_config, elf, stdin, output_dir)
        } else if self.app_only {
            run_default_app_benchmark(vm_config, elf, stdin)
        } else if self.warp {
            return run_default_native_warp_benchmark(vm_config, elf, stdin);
        } else if self.evm {
            #[cfg(feature = "evm")]
            return run_evm_benchmark(
                vm_config,
                elf,
                stdin,
                self.halo2_wrapper_k,
                self.kzg_params_dir.clone(),
            );
            #[cfg(not(feature = "evm"))]
            eyre::bail!("--evm requires the `evm` feature flag")
        } else {
            run_default_benchmark(vm_config, elf, stdin)
        }
    }
}

/// The rejected History-v4/product benchmark is intentionally uncallable.
///
/// The `--warp` CLI name is reserved for the fixed verifier-PESAT/high-arity
/// implementation so stale benchmark scripts cannot silently measure a
/// different protocol.
pub fn run_native_warp_benchmark(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
    app_params: SystemParams,
) -> eyre::Result<()> {
    let _ = (vm_config, elf, stdin, app_params);
    eyre::bail!(
        "the rejected History-v4/product WARP benchmark is retired; the fixed \
         verifier-PESAT/high-arity lane is not wired yet"
    )
}

pub fn run_benchmark(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
    app_params: SystemParams,
    leaf_params: SystemParams,
    internal_params: SystemParams,
) -> eyre::Result<()> {
    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<_> {
        let exe = VmExe::from_elf(elf, vm_config.transpiler())?;
        let app_config = AppConfig::new(vm_config, app_params);
        let agg_params = AggregationSystemParams {
            leaf: leaf_params,
            internal: internal_params,
        };
        let sdk = Sdk::new(app_config, agg_params)?;
        let lifecycle = BenchmarkLifecycle::from_env()?;
        // `Sdk` builds the app and aggregation proving keys lazily inside
        // `prove`. Forcing both `OnceLock`s here is what makes the recursive
        // lane's warm run comparable to native WARP's `prepare_history_keys`.
        let setup_start = Instant::now();
        if lifecycle.is_warm() {
            let _ = sdk.app_pk();
            let _ = sdk.agg_prover();
        }
        let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;
        reset_pcie_traffic();
        let prove_start = Instant::now();
        let (proof, baseline) = sdk.prove(exe, stdin, &[])?;
        let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;
        report_lifecycle(lifecycle, setup_ms, prove_ms);
        #[cfg(feature = "metrics")]
        {
            use openvm_stark_backend::codec::Encode;
            let serialize_start = Instant::now();
            let encoded = proof.encode_to_vec()?;
            let compressed = zstd::encode_all(&encoded[..], 19)?;
            let serialize_ms = serialize_start.elapsed().as_secs_f64() * 1000.0;
            tracing::info!(
                "Proof Size (bytes): {}, Compressed Size: {}",
                encoded.len(),
                compressed.len()
            );
            tracing::info!("Benchmark Proof Serialize Time (ms): {serialize_ms:.3}");
            metrics::gauge!("proof_size_bytes.total").set(encoded.len() as f64);
            metrics::gauge!("proof_size_bytes.compressed").set(compressed.len() as f64);
            metrics::gauge!("benchmark_proof_serialize_time_ms").set(serialize_ms);
        }
        let vk = VmStarkVerifyingKey {
            mvk: (*sdk.agg_vk()).clone(),
            baseline,
        };
        measure_verify(|| verify_vm_stark_proof_decoded(&vk, &proof).map_err(eyre::Report::from))?;
        Ok(())
    })
}

const SEGMENT_CAPTURE_FORMAT: &str = "openvm-raw-segment-proofs";
const SEGMENT_CAPTURE_VERSION: u32 = 1;
const SEGMENT_CAPTURE_ZSTD_LEVEL: i32 = 3;
const SEGMENT_CAPTURE_MAX_PROOFS: usize = 1 << 20;
const SEGMENT_CAPTURE_MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;
const SEGMENT_CAPTURE_MANIFEST_FILE: &str = "manifest.json";
const SEGMENT_CAPTURE_APP_VK_FILE: &str = "app-vk.bitcode.zst";
const SEGMENT_CAPTURE_PUBLIC_VALUES_FILE: &str = "user-public-values.bin.zst";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SegmentCaptureManifest {
    format: String,
    version: u32,
    proof_codec: String,
    segment_count: usize,
    app_vk_file: String,
    app_vk_uncompressed_bytes: usize,
    app_vk_compressed_bytes: u64,
    app_vk_bitcode_sha256: String,
    app_exe_commit: String,
    user_public_values_file: String,
    user_public_values_uncompressed_bytes: usize,
    user_public_values_compressed_bytes: u64,
    user_public_values_sha256: String,
    proofs: Vec<SegmentCaptureEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SegmentCaptureEntry {
    segment_index: usize,
    file: String,
    canonical_bytes: usize,
    compressed_bytes: u64,
    canonical_sha256: String,
    shape: SegmentProofShapeSummary,
}

/// Compact, value-independent description of every dimension that the fixed
/// verifier relation must accept. `shape_sha256` additionally commits to all
/// nested vector lengths, including every WHIR query and Merkle path, without
/// making the manifest hundreds of megabytes larger.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SegmentProofShapeSummary {
    shape_sha256: String,
    num_airs: usize,
    active_air_ids: Vec<usize>,
    active_log_heights: Vec<usize>,
    cached_commitments_per_active_air: Vec<usize>,
    public_values_per_air: Vec<usize>,
    gkr_claim_layers: usize,
    gkr_sumcheck_rounds: usize,
    batch_sumcheck_rounds: usize,
    batch_column_openings: usize,
    stacking_sumcheck_rounds: usize,
    stacking_commit_widths: Vec<usize>,
    whir_sumcheck_rounds: usize,
    whir_codeword_commits: usize,
    whir_initial_queries_per_commit: Vec<usize>,
    whir_initial_fold_rows_per_commit: Vec<usize>,
    whir_initial_widths_per_commit: Vec<usize>,
    whir_initial_max_merkle_depth: usize,
    whir_codeword_query_rounds: usize,
    whir_codeword_max_merkle_depth: usize,
    whir_final_poly_len: usize,
}

fn append_shape_len(fingerprint: &mut Vec<u8>, value: usize) {
    fingerprint.extend_from_slice(&(value as u64).to_le_bytes());
}

fn segment_proof_shape(proof: &Proof<SC>) -> SegmentProofShapeSummary {
    let mut fingerprint = Vec::new();
    append_shape_len(&mut fingerprint, proof.trace_vdata.len());
    for vdata in &proof.trace_vdata {
        match vdata {
            None => append_shape_len(&mut fingerprint, 0),
            Some(vdata) => {
                append_shape_len(&mut fingerprint, 1);
                append_shape_len(&mut fingerprint, vdata.log_height);
                append_shape_len(&mut fingerprint, vdata.cached_commitments.len());
            }
        }
    }
    append_shape_len(&mut fingerprint, proof.public_values.len());
    for values in &proof.public_values {
        append_shape_len(&mut fingerprint, values.len());
    }

    let gkr = &proof.gkr_proof;
    append_shape_len(&mut fingerprint, gkr.claims_per_layer.len());
    append_shape_len(&mut fingerprint, gkr.sumcheck_polys.len());
    for round in &gkr.sumcheck_polys {
        append_shape_len(&mut fingerprint, round.len());
    }

    let batch = &proof.batch_constraint_proof;
    append_shape_len(&mut fingerprint, batch.numerator_term_per_air.len());
    append_shape_len(&mut fingerprint, batch.denominator_term_per_air.len());
    append_shape_len(&mut fingerprint, batch.univariate_round_coeffs.len());
    append_shape_len(&mut fingerprint, batch.sumcheck_round_polys.len());
    for round in &batch.sumcheck_round_polys {
        append_shape_len(&mut fingerprint, round.len());
    }
    append_shape_len(&mut fingerprint, batch.column_openings.len());
    for air in &batch.column_openings {
        append_shape_len(&mut fingerprint, air.len());
        for part in air {
            append_shape_len(&mut fingerprint, part.len());
        }
    }

    let stacking = &proof.stacking_proof;
    append_shape_len(&mut fingerprint, stacking.univariate_round_coeffs.len());
    append_shape_len(&mut fingerprint, stacking.sumcheck_round_polys.len());
    append_shape_len(&mut fingerprint, stacking.stacking_openings.len());
    for opening in &stacking.stacking_openings {
        append_shape_len(&mut fingerprint, opening.len());
    }

    let whir = &proof.whir_proof;
    append_shape_len(&mut fingerprint, whir.whir_sumcheck_polys.len());
    append_shape_len(&mut fingerprint, whir.codeword_commits.len());
    append_shape_len(&mut fingerprint, whir.ood_values.len());
    append_shape_len(&mut fingerprint, whir.folding_pow_witnesses.len());
    append_shape_len(&mut fingerprint, whir.query_phase_pow_witnesses.len());
    append_shape_len(&mut fingerprint, whir.initial_round_opened_rows.len());
    for commit in &whir.initial_round_opened_rows {
        append_shape_len(&mut fingerprint, commit.len());
        for query in commit {
            append_shape_len(&mut fingerprint, query.len());
            for row in query {
                append_shape_len(&mut fingerprint, row.len());
            }
        }
    }
    append_shape_len(&mut fingerprint, whir.initial_round_merkle_proofs.len());
    for commit in &whir.initial_round_merkle_proofs {
        append_shape_len(&mut fingerprint, commit.len());
        for path in commit {
            append_shape_len(&mut fingerprint, path.len());
        }
    }
    append_shape_len(&mut fingerprint, whir.codeword_opened_values.len());
    for round in &whir.codeword_opened_values {
        append_shape_len(&mut fingerprint, round.len());
        for query in round {
            append_shape_len(&mut fingerprint, query.len());
        }
    }
    append_shape_len(&mut fingerprint, whir.codeword_merkle_proofs.len());
    for round in &whir.codeword_merkle_proofs {
        append_shape_len(&mut fingerprint, round.len());
        for path in round {
            append_shape_len(&mut fingerprint, path.len());
        }
    }
    append_shape_len(&mut fingerprint, whir.final_poly.len());

    let active = proof
        .trace_vdata
        .iter()
        .enumerate()
        .filter_map(|(air_id, vdata)| vdata.as_ref().map(|vdata| (air_id, vdata)))
        .collect::<Vec<_>>();
    let batch_column_openings = batch
        .column_openings
        .iter()
        .flat_map(|air| air.iter())
        .map(Vec::len)
        .sum();
    let initial_queries_per_commit = whir
        .initial_round_opened_rows
        .iter()
        .map(Vec::len)
        .collect();
    let initial_fold_rows_per_commit = whir
        .initial_round_opened_rows
        .iter()
        .map(|commit| commit.first().map_or(0, Vec::len))
        .collect();
    let initial_widths_per_commit = whir
        .initial_round_opened_rows
        .iter()
        .map(|commit| {
            commit
                .first()
                .and_then(|query| query.first())
                .map_or(0, Vec::len)
        })
        .collect();
    let initial_max_merkle_depth = whir
        .initial_round_merkle_proofs
        .iter()
        .flat_map(|commit| commit.iter())
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    let codeword_max_merkle_depth = whir
        .codeword_merkle_proofs
        .iter()
        .flat_map(|round| round.iter())
        .map(Vec::len)
        .max()
        .unwrap_or(0);

    SegmentProofShapeSummary {
        shape_sha256: sha256_hex(&fingerprint),
        num_airs: proof.trace_vdata.len(),
        active_air_ids: active.iter().map(|(air_id, _)| *air_id).collect(),
        active_log_heights: active.iter().map(|(_, vdata)| vdata.log_height).collect(),
        cached_commitments_per_active_air: active
            .iter()
            .map(|(_, vdata)| vdata.cached_commitments.len())
            .collect(),
        public_values_per_air: proof.public_values.iter().map(Vec::len).collect(),
        gkr_claim_layers: gkr.claims_per_layer.len(),
        gkr_sumcheck_rounds: gkr.sumcheck_polys.len(),
        batch_sumcheck_rounds: batch.sumcheck_round_polys.len(),
        batch_column_openings,
        stacking_sumcheck_rounds: stacking.sumcheck_round_polys.len(),
        stacking_commit_widths: stacking.stacking_openings.iter().map(Vec::len).collect(),
        whir_sumcheck_rounds: whir.whir_sumcheck_polys.len(),
        whir_codeword_commits: whir.codeword_commits.len(),
        whir_initial_queries_per_commit: initial_queries_per_commit,
        whir_initial_fold_rows_per_commit: initial_fold_rows_per_commit,
        whir_initial_widths_per_commit: initial_widths_per_commit,
        whir_initial_max_merkle_depth: initial_max_merkle_depth,
        whir_codeword_query_rounds: whir.codeword_opened_values.len(),
        whir_codeword_max_merkle_depth: codeword_max_merkle_depth,
        whir_final_poly_len: whir.final_poly.len(),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    bytes_hex(&digest)
}

fn bytes_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String is infallible");
    }
    out
}

fn create_new_capture_dir(output_dir: &Path) -> eyre::Result<()> {
    if output_dir.as_os_str().is_empty() {
        eyre::bail!("segment proof capture directory must not be empty");
    }
    if let Some(parent) = output_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(output_dir).map_err(|error| {
        eyre::eyre!(
            "cannot create fresh segment proof capture directory {}: {error}",
            output_dir.display()
        )
    })?;
    Ok(())
}

fn write_zstd_new(output_dir: &Path, file_name: &str, bytes: &[u8]) -> eyre::Result<u64> {
    let final_path = output_dir.join(file_name);
    let partial_path = output_dir.join(format!(".{file_name}.partial"));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial_path)?;
    let mut encoder = zstd::stream::write::Encoder::new(file, SEGMENT_CAPTURE_ZSTD_LEVEL)?;
    encoder.write_all(bytes)?;
    let file = encoder.finish()?;
    file.sync_all()?;
    fs::rename(&partial_path, &final_path)?;
    Ok(fs::metadata(final_path)?.len())
}

fn write_manifest_new(output_dir: &Path, manifest: &SegmentCaptureManifest) -> eyre::Result<()> {
    let bytes = serde_json::to_vec_pretty(manifest)?;
    let final_path = output_dir.join(SEGMENT_CAPTURE_MANIFEST_FILE);
    let partial_path = output_dir.join(".manifest.json.partial");
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial_path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(partial_path, final_path)?;
    File::open(output_dir)?.sync_all()?;
    Ok(())
}

fn read_zstd_exact(path: &Path, expected_len: usize) -> eyre::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut decoder = zstd::stream::read::Decoder::new(file)?;
    let limit = u64::try_from(expected_len)?
        .checked_add(1)
        .ok_or_else(|| eyre::eyre!("capture decompression bound overflow"))?;
    let mut bytes = Vec::with_capacity(expected_len);
    decoder.by_ref().take(limit).read_to_end(&mut bytes)?;
    if bytes.len() != expected_len {
        eyre::bail!(
            "captured object {} decoded to {} bytes, expected {}",
            path.display(),
            bytes.len(),
            expected_len
        );
    }
    Ok(bytes)
}

fn is_safe_capture_file_name(file_name: &str) -> bool {
    let mut components = Path::new(file_name).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

fn expected_proof_codec_label() -> String {
    format!("openvm-stark-proof-codec-v{PROOF_CODEC_VERSION}")
}

fn validate_capture_manifest(
    output_dir: &Path,
    manifest: &SegmentCaptureManifest,
) -> eyre::Result<()> {
    if manifest.format != SEGMENT_CAPTURE_FORMAT
        || manifest.version != SEGMENT_CAPTURE_VERSION
        || manifest.proof_codec != expected_proof_codec_label()
    {
        eyre::bail!("unsupported or mismatched segment-proof capture format");
    }
    if manifest.segment_count != manifest.proofs.len()
        || manifest.segment_count > SEGMENT_CAPTURE_MAX_PROOFS
    {
        eyre::bail!("invalid segment count in capture manifest");
    }
    if manifest.app_vk_file != SEGMENT_CAPTURE_APP_VK_FILE
        || manifest.user_public_values_file != SEGMENT_CAPTURE_PUBLIC_VALUES_FILE
    {
        eyre::bail!("capture manifest substitutes a fixed metadata file");
    }

    let mut expected_files = BTreeSet::from([
        SEGMENT_CAPTURE_MANIFEST_FILE.to_owned(),
        manifest.app_vk_file.clone(),
        manifest.user_public_values_file.clone(),
    ]);
    for (expected_index, entry) in manifest.proofs.iter().enumerate() {
        let expected_file = format!("segment-{expected_index:06}.proof.bin.zst");
        if entry.segment_index != expected_index
            || entry.file != expected_file
            || !is_safe_capture_file_name(&entry.file)
            || !expected_files.insert(entry.file.clone())
        {
            eyre::bail!("invalid, reordered, or duplicate segment entry {expected_index}");
        }
    }

    let mut actual_files = BTreeSet::new();
    for entry in fs::read_dir(output_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            eyre::bail!("capture directory contains a non-file artifact");
        }
        actual_files.insert(entry.file_name().to_string_lossy().into_owned());
    }
    if actual_files != expected_files {
        eyre::bail!("capture directory contains missing, extra, or partial artifacts");
    }
    Ok(())
}

fn validate_compressed_len(path: &Path, expected_len: u64) -> eyre::Result<()> {
    let actual_len = fs::metadata(path)?.len();
    if actual_len != expected_len {
        eyre::bail!(
            "captured object {} has compressed length {}, expected {}",
            path.display(),
            actual_len,
            expected_len
        );
    }
    Ok(())
}

fn decode_verified_segment_capture(
    output_dir: &Path,
    manifest: &SegmentCaptureManifest,
) -> eyre::Result<(AppVerifyingKey, ContinuationVmProof<SC>, Vec<u8>)> {
    validate_capture_manifest(output_dir, manifest)?;

    let app_vk_path = output_dir.join(&manifest.app_vk_file);
    validate_compressed_len(&app_vk_path, manifest.app_vk_compressed_bytes)?;
    let app_vk_bytes = read_zstd_exact(&app_vk_path, manifest.app_vk_uncompressed_bytes)?;
    if sha256_hex(&app_vk_bytes) != manifest.app_vk_bitcode_sha256 {
        eyre::bail!("captured app verifying key failed length/digest validation");
    }
    let app_vk: AppVerifyingKey = bitcode::deserialize(&app_vk_bytes)?;

    let mut per_segment = Vec::with_capacity(manifest.proofs.len());
    for entry in &manifest.proofs {
        let proof_path = output_dir.join(&entry.file);
        validate_compressed_len(&proof_path, entry.compressed_bytes)?;
        let bytes = read_zstd_exact(&proof_path, entry.canonical_bytes)?;
        if sha256_hex(&bytes) != entry.canonical_sha256 {
            eyre::bail!(
                "captured segment {} failed length/digest validation",
                entry.segment_index
            );
        }
        let mut reader = bytes.as_slice();
        let decoded = Proof::<SC>::decode(&mut reader)?;
        if !reader.is_empty() {
            eyre::bail!(
                "captured segment {} has {} trailing canonical bytes",
                entry.segment_index,
                reader.len()
            );
        }
        if segment_proof_shape(&decoded) != entry.shape {
            eyre::bail!("captured segment {} changed shape", entry.segment_index);
        }
        per_segment.push(decoded);
    }

    let pvs_path = output_dir.join(&manifest.user_public_values_file);
    validate_compressed_len(&pvs_path, manifest.user_public_values_compressed_bytes)?;
    let pvs_bytes = read_zstd_exact(&pvs_path, manifest.user_public_values_uncompressed_bytes)?;
    if sha256_hex(&pvs_bytes) != manifest.user_public_values_sha256 {
        eyre::bail!("captured user public values failed length/digest validation");
    }
    let mut pvs_reader = pvs_bytes.as_slice();
    let user_public_values =
        UserPublicValuesProof::<DIGEST_WIDTH, F>::decode::<SC, _>(&mut pvs_reader)?;
    if !pvs_reader.is_empty() {
        eyre::bail!(
            "captured user public values have {} trailing bytes",
            pvs_reader.len()
        );
    }

    let continuation = ContinuationVmProof {
        per_segment,
        user_public_values,
    };
    let reloaded_commit = verify_app_proof::<DefaultStarkEngine>(&app_vk, &continuation)?;
    let mut reloaded_commit_bytes = Vec::new();
    <SC as EncodableConfig>::encode_digest(&reloaded_commit, &mut reloaded_commit_bytes)?;
    Ok((app_vk, continuation, reloaded_commit_bytes))
}

fn verify_written_segment_capture(
    output_dir: &Path,
    manifest: &SegmentCaptureManifest,
    expected_app_exe_commit: &[u8],
) -> eyre::Result<()> {
    if manifest.app_exe_commit != bytes_hex(expected_app_exe_commit) {
        eyre::bail!("capture manifest binds a different app executable commitment");
    }
    let (_, _, reloaded_commit_bytes) = decode_verified_segment_capture(output_dir, manifest)?;
    if reloaded_commit_bytes != expected_app_exe_commit {
        eyre::bail!("reloaded capture verifies to a different app executable commitment");
    }
    Ok(())
}

/// Decode and independently verify a versioned raw SWIRL segment-proof
/// capture before returning it to an aggregation benchmark.
///
/// File names, counts, compressed and canonical lengths, SHA-256 digests,
/// proof shapes, segment order, application VK, public values, and the final
/// executable commitment are all checked. No captured proof is exposed until
/// the ordinary application-proof verifier accepts the complete sequence.
pub fn read_verified_segment_proof_capture(
    output_dir: &Path,
) -> eyre::Result<(AppVerifyingKey, ContinuationVmProof<SC>)> {
    let manifest_path = output_dir.join(SEGMENT_CAPTURE_MANIFEST_FILE);
    let metadata = fs::symlink_metadata(&manifest_path)?;
    if !metadata.file_type().is_file() || metadata.len() > SEGMENT_CAPTURE_MAX_MANIFEST_BYTES {
        eyre::bail!("segment capture manifest is not a bounded regular file");
    }
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: SegmentCaptureManifest = serde_json::from_slice(&manifest_bytes)?;
    let expected_commitment = manifest.app_exe_commit.clone();
    let (app_vk, continuation, commitment_bytes) =
        decode_verified_segment_capture(output_dir, &manifest)?;
    if bytes_hex(&commitment_bytes) != expected_commitment {
        eyre::bail!("reloaded capture verifies to a different app executable commitment");
    }
    Ok((app_vk, continuation))
}

/// Write a fresh, versioned capture of an already-produced continuation proof.
/// The ordinary native verifier is run before any artifact is trusted and once
/// again after every compressed object has been decoded from disk.
pub fn write_verified_segment_proof_capture(
    app_vk: &AppVerifyingKey,
    continuation: &ContinuationVmProof<SC>,
    output_dir: &Path,
) -> eyre::Result<()> {
    create_new_capture_dir(output_dir)?;
    let app_exe_commit = verify_app_proof::<DefaultStarkEngine>(app_vk, continuation)?;
    let app_vk_bytes = bitcode::serialize(app_vk)?;
    let app_vk_compressed_bytes =
        write_zstd_new(output_dir, SEGMENT_CAPTURE_APP_VK_FILE, &app_vk_bytes)?;
    let mut app_exe_commit_bytes = Vec::new();
    <SC as EncodableConfig>::encode_digest(&app_exe_commit, &mut app_exe_commit_bytes)?;

    let mut pvs_bytes = Vec::new();
    continuation
        .user_public_values
        .encode::<SC, _>(&mut pvs_bytes)?;
    let pvs_compressed_bytes =
        write_zstd_new(output_dir, SEGMENT_CAPTURE_PUBLIC_VALUES_FILE, &pvs_bytes)?;

    let mut entries = Vec::with_capacity(continuation.per_segment.len());
    for (segment_index, proof) in continuation.per_segment.iter().enumerate() {
        let canonical = proof.encode_to_vec()?;
        let file = format!("segment-{segment_index:06}.proof.bin.zst");
        let compressed_bytes = write_zstd_new(output_dir, &file, &canonical)?;
        entries.push(SegmentCaptureEntry {
            segment_index,
            file,
            canonical_bytes: canonical.len(),
            compressed_bytes,
            canonical_sha256: sha256_hex(&canonical),
            shape: segment_proof_shape(proof),
        });
    }

    let manifest = SegmentCaptureManifest {
        format: SEGMENT_CAPTURE_FORMAT.to_owned(),
        version: SEGMENT_CAPTURE_VERSION,
        proof_codec: expected_proof_codec_label(),
        segment_count: entries.len(),
        app_vk_file: SEGMENT_CAPTURE_APP_VK_FILE.to_owned(),
        app_vk_uncompressed_bytes: app_vk_bytes.len(),
        app_vk_compressed_bytes,
        app_vk_bitcode_sha256: sha256_hex(&app_vk_bytes),
        app_exe_commit: bytes_hex(&app_exe_commit_bytes),
        user_public_values_file: SEGMENT_CAPTURE_PUBLIC_VALUES_FILE.to_owned(),
        user_public_values_uncompressed_bytes: pvs_bytes.len(),
        user_public_values_compressed_bytes: pvs_compressed_bytes,
        user_public_values_sha256: sha256_hex(&pvs_bytes),
        proofs: entries,
    };
    write_manifest_new(output_dir, &manifest)?;
    let manifest_bytes = fs::read(output_dir.join(SEGMENT_CAPTURE_MANIFEST_FILE))?;
    let reloaded_manifest: SegmentCaptureManifest = serde_json::from_slice(&manifest_bytes)?;
    if reloaded_manifest != manifest {
        eyre::bail!("reloaded capture manifest differs from the manifest that was written");
    }
    verify_written_segment_capture(output_dir, &reloaded_manifest, &app_exe_commit_bytes)
}

/// Generates the ordinary SWIRL continuation proofs and captures them before
/// recursive aggregation. This is deliberately an explicit benchmark command:
/// generic proving APIs never acquire filesystem side effects from profiling.
pub fn run_segment_proof_capture(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
    app_params: SystemParams,
    output_dir: &Path,
) -> eyre::Result<()> {
    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<_> {
        let exe = VmExe::from_elf(elf, vm_config.transpiler())?;
        let app_config = AppConfig::new(vm_config, app_params);
        let sdk = Sdk::new(app_config, Default::default())?;
        let (_, app_vk) = sdk.app_keygen();
        let mut prover = sdk.app_prover(exe)?;

        let prove_start = Instant::now();
        let continuation = prover.prove(stdin)?;
        let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;
        let capture_start = Instant::now();
        write_verified_segment_proof_capture(&app_vk, &continuation, output_dir)?;
        let capture_ms = capture_start.elapsed().as_secs_f64() * 1000.0;

        tracing::info!(
            segment_count = continuation.per_segment.len(),
            prove_ms,
            capture_and_double_verify_ms = capture_ms,
            capture_dir = %output_dir.display(),
            "captured and re-verified raw SWIRL segment proofs"
        );
        Ok(())
    })
}

pub fn run_default_segment_proof_capture(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
    output_dir: &Path,
) -> eyre::Result<()> {
    run_segment_proof_capture(
        vm_config,
        elf,
        stdin,
        default_bench_app_params(),
        output_dir,
    )
}

pub fn run_app_benchmark(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
    app_params: SystemParams,
) -> eyre::Result<()> {
    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<_> {
        let exe = VmExe::from_elf(elf, vm_config.transpiler())?;
        let app_config = AppConfig::new(vm_config, app_params);
        let sdk = Sdk::new(app_config, Default::default())?;
        let (_, app_vk) = sdk.app_keygen();
        let mut prover = sdk.app_prover(exe)?;
        let proof = prover.prove(stdin)?;
        let _ = verify_app_proof::<DefaultStarkEngine>(&app_vk, &proof)?;
        Ok(())
    })
}

pub fn run_default_benchmark(vm_config: SdkVmConfig, elf: Elf, stdin: StdIn) -> eyre::Result<()> {
    run_benchmark(
        vm_config,
        elf,
        stdin,
        default_bench_app_params(),
        leaf_params_with_100_bits_security(),
        internal_params_with_100_bits_security(),
    )
}

pub fn run_default_native_warp_benchmark(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
) -> eyre::Result<()> {
    run_native_warp_benchmark(vm_config, elf, stdin, default_warp_bench_app_params())
}

pub fn run_default_app_benchmark(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
) -> eyre::Result<()> {
    run_app_benchmark(vm_config, elf, stdin, default_bench_app_params())
}

#[cfg(feature = "evm")]
pub fn run_evm_benchmark(
    vm_config: SdkVmConfig,
    elf: Elf,
    stdin: StdIn,
    halo2_wrapper_k: Option<usize>,
    kzg_params_dir: Option<std::path::PathBuf>,
) -> eyre::Result<()> {
    use openvm_sdk::config::Halo2Config;

    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<_> {
        let exe = VmExe::from_elf(elf, vm_config.transpiler())?;
        let app_config = AppConfig::new(vm_config, default_bench_app_params());
        let agg_params = AggregationSystemParams {
            leaf: leaf_params_with_100_bits_security(),
            internal: internal_params_with_100_bits_security(),
        };
        let mut builder = Sdk::builder().app_config(app_config).agg_params(agg_params);
        if halo2_wrapper_k.is_some() {
            builder = builder.halo2_config(
                Default::default(),
                Halo2Config {
                    wrapper_k: halo2_wrapper_k,
                    profiling: false,
                },
            );
        }
        if let Some(dir) = kzg_params_dir {
            builder = builder.halo2_params_dir(dir);
        }
        let sdk = builder.build()?;
        let evm_proof = sdk.prove_evm(exe, stdin, &[])?;
        let verifier = sdk.generate_halo2_verifier_solidity()?;
        let gas_cost = Sdk::verify_evm_halo2_proof(&verifier, evm_proof, None)?;
        tracing::info!("EVM verification gas cost: {gas_cost}");
        Ok(())
    })
}

pub fn default_bench_app_params() -> SystemParams {
    app_params_with_100_bits_security(DEFAULT_LOG_STACKED_HEIGHT)
}

/// The WARP lane's application profile.
///
/// Identical to [`default_bench_app_params`] in every parameter the comparison is
/// about -- rate, fold arity, query counts, all three grinding budgets, security
/// target, LogUp -- and different in exactly one: the MMCS row grouping used when
/// committing a segment's main trace.
///
/// That one is not a security or performance parameter, it is a commitment layout.
/// The recursive lane folds the application root with WHIR, so its root must group
/// `2^k` rows per Merkle leaf, since a fold round reads its values out of one leaf.
/// The WARP lane never folds it: a segment carries no WHIR proof at all, and the root
/// is read only by single-point shift queries. Forcing the same layout on it would
/// make every query reveal and re-hash `2^k` times more cells for a fold that does
/// not happen.
///
/// `comparison_profile_parity` asserts the equality of everything else, and records
/// that this field is deliberately allowed to differ.
pub fn default_warp_bench_app_params() -> SystemParams {
    native_warp_app_params_with_100_bits_security(DEFAULT_LOG_STACKED_HEIGHT)
}

#[cfg(test)]
mod segment_capture_tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn fresh_test_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "openvm-segment-capture-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn empty_shape() -> SegmentProofShapeSummary {
        SegmentProofShapeSummary {
            shape_sha256: "00".repeat(32),
            num_airs: 0,
            active_air_ids: Vec::new(),
            active_log_heights: Vec::new(),
            cached_commitments_per_active_air: Vec::new(),
            public_values_per_air: Vec::new(),
            gkr_claim_layers: 0,
            gkr_sumcheck_rounds: 0,
            batch_sumcheck_rounds: 0,
            batch_column_openings: 0,
            stacking_sumcheck_rounds: 0,
            stacking_commit_widths: Vec::new(),
            whir_sumcheck_rounds: 0,
            whir_codeword_commits: 0,
            whir_initial_queries_per_commit: Vec::new(),
            whir_initial_fold_rows_per_commit: Vec::new(),
            whir_initial_widths_per_commit: Vec::new(),
            whir_initial_max_merkle_depth: 0,
            whir_codeword_query_rounds: 0,
            whir_codeword_max_merkle_depth: 0,
            whir_final_poly_len: 0,
        }
    }

    fn one_proof_manifest(app_commit: &[u8]) -> SegmentCaptureManifest {
        SegmentCaptureManifest {
            format: SEGMENT_CAPTURE_FORMAT.to_owned(),
            version: SEGMENT_CAPTURE_VERSION,
            proof_codec: expected_proof_codec_label(),
            segment_count: 1,
            app_vk_file: SEGMENT_CAPTURE_APP_VK_FILE.to_owned(),
            app_vk_uncompressed_bytes: 0,
            app_vk_compressed_bytes: 0,
            app_vk_bitcode_sha256: "00".repeat(32),
            app_exe_commit: bytes_hex(app_commit),
            user_public_values_file: SEGMENT_CAPTURE_PUBLIC_VALUES_FILE.to_owned(),
            user_public_values_uncompressed_bytes: 0,
            user_public_values_compressed_bytes: 0,
            user_public_values_sha256: "00".repeat(32),
            proofs: vec![SegmentCaptureEntry {
                segment_index: 0,
                file: "segment-000000.proof.bin.zst".to_owned(),
                canonical_bytes: 0,
                compressed_bytes: 0,
                canonical_sha256: "00".repeat(32),
                shape: empty_shape(),
            }],
        }
    }

    #[test]
    fn capture_directory_is_create_new_and_decompression_is_bounded() {
        let directory = fresh_test_dir("bounded");
        create_new_capture_dir(&directory).unwrap();
        assert!(create_new_capture_dir(&directory).is_err());
        write_zstd_new(&directory, "bounded.zst", b"abcd").unwrap();
        assert_eq!(
            read_zstd_exact(&directory.join("bounded.zst"), 4).unwrap(),
            b"abcd"
        );
        assert!(read_zstd_exact(&directory.join("bounded.zst"), 3).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn manifest_rejects_reordering_substitution_and_extra_files() {
        let directory = fresh_test_dir("manifest");
        fs::create_dir(&directory).unwrap();
        for file in [
            SEGMENT_CAPTURE_MANIFEST_FILE,
            SEGMENT_CAPTURE_APP_VK_FILE,
            SEGMENT_CAPTURE_PUBLIC_VALUES_FILE,
            "segment-000000.proof.bin.zst",
        ] {
            fs::write(directory.join(file), []).unwrap();
        }
        let app_commit = [1u8, 2, 3, 4];
        let manifest = one_proof_manifest(&app_commit);
        validate_capture_manifest(&directory, &manifest, &app_commit).unwrap();

        let mut reordered = manifest.clone();
        reordered.proofs[0].segment_index = 1;
        assert!(validate_capture_manifest(&directory, &reordered, &app_commit).is_err());

        let mut substituted = manifest.clone();
        substituted.proofs[0].file = "../segment-000000.proof.bin.zst".to_owned();
        assert!(validate_capture_manifest(&directory, &substituted, &app_commit).is_err());

        fs::write(directory.join(".partial"), []).unwrap();
        assert!(validate_capture_manifest(&directory, &manifest, &app_commit).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn configured_real_capture_reloads_and_verifies() {
        let Some(directory) = std::env::var_os("OPENVM_TEST_SEGMENT_CAPTURE") else {
            return;
        };
        let directory = PathBuf::from(directory);
        let manifest_bytes = fs::read(directory.join(SEGMENT_CAPTURE_MANIFEST_FILE)).unwrap();
        let manifest: SegmentCaptureManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest.app_exe_commit.len() % 2, 0);
        let expected_app_exe_commit = manifest
            .app_exe_commit
            .as_bytes()
            .chunks_exact(2)
            .map(|digits| {
                let digits = std::str::from_utf8(digits).unwrap();
                u8::from_str_radix(digits, 16).unwrap()
            })
            .collect::<Vec<_>>();
        verify_written_segment_capture(&directory, &manifest, &expected_app_exe_commit).unwrap();
    }
}

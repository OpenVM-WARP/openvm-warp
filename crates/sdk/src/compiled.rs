#[cfg(feature = "rvr")]
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(feature = "rvr")]
use eyre::{Context, Result};
use openvm_circuit::arch::execution_mode::{MeteredCostCtx, MeteredCtx};
#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::{execution_mode::ExecutionCtx, InterpretedInstance};
#[cfg(feature = "rvr")]
use openvm_circuit::arch::{
    execution_mode::{MeteredCtxConfig, SegmentationConfig},
    instructions::exe::VmExe,
    rvr::{runtime_toolchain, CompileError},
};
#[cfg(feature = "rvr")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "rvr")]
use sha2::{Digest as _, Sha256};

use crate::F;

cfg_if::cfg_if! {
    if #[cfg(feature = "rvr")] {
        use openvm_circuit::arch::rvr::{
            RvrMeteredCostInstance, RvrMeteredInstance, RvrPureInstance,
        };
        pub type CompiledExePure<'a, F> = RvrPureInstance<'a, F>;
        pub type MeteredInstance<'a, F> = RvrMeteredInstance<'a, F>;
        pub type MeteredCostInstance<'a, F> = RvrMeteredCostInstance<'a, F>;
    } else if #[cfg(feature = "aot")] {
        use openvm_circuit::arch::AotInstance;
        pub type CompiledExePure<'a, F> = AotInstance<'a, F, ExecutionCtx>;
        pub type MeteredInstance<'a, F> = AotInstance<'a, F, MeteredCtx>;
        // AOT has no dedicated metered-cost backend; fall back to the interpreter.
        pub type MeteredCostInstance<'a, F> = InterpretedInstance<'a, F, MeteredCostCtx>;
    } else {
        pub type CompiledExePure<'a, F> = InterpretedInstance<'a, F, ExecutionCtx>;
        pub type MeteredInstance<'a, F> = InterpretedInstance<'a, F, MeteredCtx>;
        pub type MeteredCostInstance<'a, F> = InterpretedInstance<'a, F, MeteredCostCtx>;
    }
}

/// Bundles a [`MeteredInstance`] with a precomputed [`MeteredCtx`] so each execution
/// just clones the ctx instead of rebuilding from the proving key.
pub struct CompiledExeMetered<'a> {
    pub instance: MeteredInstance<'a, F>,
    pub ctx: MeteredCtx,
    #[cfg(feature = "rvr")]
    pub executor_idx_to_air_idx: Vec<usize>,
    #[cfg(feature = "rvr")]
    pub(crate) artifact_identity: MeteredArtifactIdentity,
}

pub struct CompiledExeMeteredCost<'a> {
    pub instance: MeteredCostInstance<'a, F>,
    pub ctx: MeteredCostCtx,
}

#[cfg(feature = "rvr")]
pub const METERED_ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Inputs that bind a cached native metered executable to the executable, VM shape, and native
/// toolchain that produced it. This protects against stale or accidentally mismatched cache
/// entries. The cache directory itself remains a trusted local deployment boundary.
#[cfg(feature = "rvr")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeteredArtifactIdentity {
    pub format_version: u32,
    pub openvm_version: String,
    pub metered_abi_version: u32,
    pub target_arch: String,
    pub target_os: String,
    pub compiler: String,
    pub compiler_identity_sha256: String,
    pub linker: String,
    pub linker_identity_sha256: String,
    pub make: String,
    pub make_identity_sha256: String,
    pub executable_sha256: String,
    pub metering_shape_sha256: String,
}

#[cfg(feature = "rvr")]
impl MeteredArtifactIdentity {
    pub fn cache_key(&self) -> Result<String> {
        sha256_serialized(self)
    }

    pub fn cache_library_path(&self, cache_dir: &Path) -> Result<PathBuf> {
        Ok(cache_dir.join(self.cache_key()?).join(format!(
            "openvm-metered.{}",
            std::env::consts::DLL_EXTENSION
        )))
    }
}

#[cfg(feature = "rvr")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeteredArtifactCacheStatus {
    Hit,
    Miss,
    Rebuilt { reason: String },
}

#[cfg(feature = "rvr")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeteredArtifactMetadata {
    pub format_version: u32,
    pub identity: MeteredArtifactIdentity,
    pub library_sha256: String,
    pub metered_ctx_config: MeteredCtxConfig,
    pub segmentation_config: SegmentationConfig,
    pub executor_idx_to_air_idx: Vec<usize>,
}

#[cfg(feature = "rvr")]
#[derive(Serialize)]
struct MeteringShape<'a> {
    metered_ctx_config: &'a MeteredCtxConfig,
    segmentation_config: &'a SegmentationConfig,
    executor_idx_to_air_idx: &'a [usize],
}

#[cfg(feature = "rvr")]
fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[cfg(feature = "rvr")]
fn sha256_serialized<T: Serialize>(value: &T) -> Result<String> {
    // `VmExe::init_memory` has tuple keys, which JSON cannot encode as object keys. Bitcode is
    // already OpenVM's canonical persisted representation for serde values and is deterministic
    // for the executable's ordered maps.
    let bytes = bitcode::serialize(value).context("failed to serialize artifact identity")?;
    Ok(sha256_bytes(&bytes))
}

#[cfg(feature = "rvr")]
fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(feature = "rvr")]
fn command_path(command: &str) -> Result<PathBuf> {
    let path = Path::new(command);
    if path.components().count() > 1 {
        return fs::canonicalize(path)
            .with_context(|| format!("failed to resolve tool executable {command}"));
    }
    let search_path = std::env::var_os("PATH")
        .ok_or_else(|| eyre::eyre!("PATH is not set while resolving tool {command}"))?;
    std::env::split_paths(&search_path)
        .map(|directory| directory.join(command))
        .find(|candidate| candidate.is_file())
        .map(fs::canonicalize)
        .transpose()
        .with_context(|| format!("failed to resolve tool executable {command}"))?
        .ok_or_else(|| eyre::eyre!("tool executable {command} was not found on PATH"))
}

#[cfg(feature = "rvr")]
fn command_identity_sha256(command: &str) -> Result<String> {
    let resolved = command_path(command)?;
    let output = Command::new(command)
        .arg("--version")
        .output()
        .with_context(|| format!("failed to query tool version for {command}"))?;
    // Some multi-call drivers (notably generic `lld`) intentionally reject `--version` while
    // printing a stable diagnostic that tells the caller to select a concrete driver. Hash the
    // resolved executable itself as the authority and include the diagnostic and exit status as
    // useful extra identity material; a non-zero version-query status is therefore harmless.
    let mut bytes = fs::read(&resolved)
        .with_context(|| format!("failed to fingerprint tool {}", resolved.display()))?;
    bytes.extend_from_slice(&output.status.code().unwrap_or(-1).to_le_bytes());
    bytes.extend_from_slice(&output.stdout);
    bytes.extend_from_slice(&output.stderr);
    Ok(sha256_bytes(&bytes))
}

#[cfg(feature = "rvr")]
pub(crate) fn metered_artifact_identity(
    exe: &VmExe<F>,
    metered_ctx_config: &MeteredCtxConfig,
    segmentation_config: &SegmentationConfig,
    executor_idx_to_air_idx: &[usize],
) -> Result<MeteredArtifactIdentity> {
    let toolchain = runtime_toolchain().map_err(|error| eyre::eyre!(error))?;
    let executable_sha256 = sha256_serialized(exe)?;
    let metering_shape_sha256 = sha256_serialized(&MeteringShape {
        metered_ctx_config,
        segmentation_config,
        executor_idx_to_air_idx,
    })?;
    Ok(MeteredArtifactIdentity {
        format_version: METERED_ARTIFACT_FORMAT_VERSION,
        openvm_version: crate::OPENVM_VERSION.to_owned(),
        // Increment whenever native metered code generation or its C/Rust ABI changes without an
        // OpenVM minor-version change.
        metered_abi_version: 2,
        target_arch: std::env::consts::ARCH.to_owned(),
        target_os: std::env::consts::OS.to_owned(),
        compiler_identity_sha256: command_identity_sha256(&toolchain.compiler)?,
        compiler: toolchain.compiler,
        linker_identity_sha256: command_identity_sha256(&toolchain.linker)?,
        linker: toolchain.linker,
        make_identity_sha256: command_identity_sha256(&toolchain.make)?,
        make: toolchain.make,
        executable_sha256,
        metering_shape_sha256,
    })
}

#[cfg(feature = "rvr")]
pub fn metered_artifact_metadata_path(lib_path: &Path) -> PathBuf {
    lib_path.with_extension("json")
}

#[cfg(feature = "rvr")]
pub fn load_metered_artifact_metadata(lib_path: &Path) -> Result<MeteredArtifactMetadata> {
    let path = metered_artifact_metadata_path(lib_path);
    let data = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("failed to parse {}", path.display()))
}

#[cfg(feature = "rvr")]
fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| eyre::eyre!("artifact path has no UTF-8 filename: {}", path.display()))?;
    let temp_path =
        path.with_file_name(format!(".{file_name}.tmp-{}-{counter}", std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)
            .with_context(|| format!("failed to create {}", temp_path.display()))?;
        file.write_all(data)
            .with_context(|| format!("failed to write {}", temp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync {}", temp_path.display()))?;
        fs::rename(&temp_path, path).with_context(|| {
            format!(
                "failed to atomically replace {} with {}",
                path.display(),
                temp_path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[cfg(feature = "rvr")]
fn write_metered_artifact_metadata(
    compiled: &CompiledExeMetered<'_>,
    lib_path: &Path,
) -> Result<()> {
    let metadata = MeteredArtifactMetadata {
        format_version: METERED_ARTIFACT_FORMAT_VERSION,
        identity: compiled.artifact_identity.clone(),
        library_sha256: sha256_file(lib_path)?,
        metered_ctx_config: compiled.ctx.config.clone(),
        segmentation_config: compiled.ctx.segmentation_ctx.config().clone(),
        executor_idx_to_air_idx: compiled.executor_idx_to_air_idx.clone(),
    };
    let metadata_path = metered_artifact_metadata_path(lib_path);
    let data = serde_json::to_vec_pretty(&metadata)?;
    atomic_write(&metadata_path, &data)
}

#[cfg(feature = "rvr")]
pub(crate) fn validate_metered_artifact(
    lib_path: &Path,
    metadata: &MeteredArtifactMetadata,
    expected_identity: &MeteredArtifactIdentity,
) -> Result<()> {
    if metadata.format_version != METERED_ARTIFACT_FORMAT_VERSION {
        eyre::bail!(
            "unsupported metered artifact format {}, expected {}",
            metadata.format_version,
            METERED_ARTIFACT_FORMAT_VERSION
        );
    }
    if metadata.identity.format_version != metadata.format_version {
        eyre::bail!("metered artifact identity/manifest version mismatch");
    }
    let encoded_shape_sha256 = sha256_serialized(&MeteringShape {
        metered_ctx_config: &metadata.metered_ctx_config,
        segmentation_config: &metadata.segmentation_config,
        executor_idx_to_air_idx: &metadata.executor_idx_to_air_idx,
    })?;
    if encoded_shape_sha256 != metadata.identity.metering_shape_sha256 {
        eyre::bail!("metered artifact metadata shape digest mismatch");
    }
    if &metadata.identity != expected_identity {
        eyre::bail!("metered artifact executable, VM shape, or toolchain changed");
    }
    let actual_library_sha256 = sha256_file(lib_path)?;
    if actual_library_sha256 != metadata.library_sha256 {
        eyre::bail!("metered artifact shared-library digest mismatch");
    }
    Ok(())
}

#[cfg(feature = "rvr")]
impl CompiledExeMetered<'_> {
    #[cfg(feature = "rvr")]
    pub fn artifact_identity(&self) -> &MeteredArtifactIdentity {
        &self.artifact_identity
    }

    /// Persist the compiled shared library and static metering metadata into `dir`.
    /// Returns the path of the copied `.so`/`.dylib`.
    pub fn save(&self, dir: &Path) -> Result<PathBuf> {
        let lib_path = self.instance.save(dir)?;
        write_metered_artifact_metadata(self, &lib_path)?;
        Ok(lib_path)
    }

    /// Persist a cache entry at an exact shared-library path.
    pub fn save_to_path(&self, lib_path: &Path) -> Result<PathBuf> {
        let lib_path = self.instance.save_to_path(lib_path)?;
        write_metered_artifact_metadata(self, &lib_path)?;
        Ok(lib_path)
    }

    /// Persist generated C sources for inspection.
    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.instance.save_generated_sources(dir)
    }
}

#[cfg(feature = "rvr")]
impl CompiledExeMeteredCost<'_> {
    /// Persist the compiled shared library into `dir`. Returns the path of
    /// the copied `.so`/`.dylib`. The `MeteredCostCtx` is not persisted — it
    /// is rebuilt on load via
    /// [`Sdk::load_compiled_metered_cost`](crate::Sdk::load_compiled_metered_cost).
    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        self.instance.save(dir)
    }
}

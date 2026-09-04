//! Reduced-SWIRL to WARP proving pipeline.
//!
//! OpenVM stops after AIR/LogUp and stacking, transfers the original
//! constrained RS commitment into WARP, and recursively compresses the
//! authenticated transition tree and terminal decision. The retired
//! PCS-opening PESAT and completed-proof accumulation prototypes are absent
//! from this module graph.

pub mod reduced_swirl_boundary;
pub mod reduced_swirl_execution_cpu;
#[cfg(feature = "cuda")]
pub mod reduced_swirl_execution_cuda;
pub mod reduced_swirl_native;
#[cfg(feature = "cuda")]
pub mod reduced_swirl_native_cuda;
pub mod reduced_swirl_params;
#[cfg(feature = "cuda")]
pub mod reduced_swirl_production_cuda;
pub mod reduced_swirl_recursive_adapter;
#[cfg(feature = "cuda")]
pub mod reduced_swirl_source_leaf;
pub mod reduced_swirl_source_receipt;
pub mod reduced_swirl_source_tree_component;
pub mod reduced_swirl_terminal_component;
#[cfg(feature = "cuda")]
pub mod reduced_swirl_transition_finalizer;
pub mod reduced_swirl_transition_leaf;
pub mod reduced_swirl_vacc_component;
pub mod reduced_swirl_wrapper_components;
pub mod reduced_swirl_wrapper_system;
#[cfg(feature = "cuda")]
pub mod reduced_swirl_wrapper_system_cuda;

#[cfg(test)]
mod cutover_tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    #[test]
    fn production_sdk_sources_cannot_reference_retired_opening_pesat() {
        let forbidden = [
            ["PointOpening", "Pesat"].concat(),
            ["OpeningPesat", "Relation"].concat(),
            ["opening_pesat", "_relation"].concat(),
        ];

        for path in rust_sources(Path::new(env!("CARGO_MANIFEST_DIR")).join("src")) {
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            for symbol in &forbidden {
                assert!(
                    !source.contains(symbol),
                    "production SDK source {} references forbidden legacy symbol {symbol}",
                    path.display()
                );
            }
        }
    }

    fn rust_sources(root: PathBuf) -> Vec<PathBuf> {
        let mut pending = vec![root];
        let mut sources = Vec::new();
        while let Some(path) = pending.pop() {
            for entry in fs::read_dir(&path)
                .unwrap_or_else(|error| panic!("read directory {}: {error}", path.display()))
            {
                let path = entry.expect("read SDK source entry").path();
                if path.is_dir() {
                    pending.push(path);
                } else if path.extension().is_some_and(|extension| extension == "rs") {
                    sources.push(path);
                }
            }
        }
        sources
    }
}

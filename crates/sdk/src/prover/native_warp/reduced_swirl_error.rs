//! Errors shared by the bounded reduced-SWIRL proof components.

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlWrapperSystemError {
    #[error("invalid reduced-SWIRL wrapper binding: {0}")]
    Binding(&'static str),
    #[error("reduced-SWIRL wrapper keygen failed: {0}")]
    Keygen(String),
    #[error("reduced-SWIRL wrapper key integrity failed: {0}")]
    KeyIntegrity(&'static str),
    #[error("reduced-SWIRL wrapper component context error: {0}")]
    Context(&'static str),
    #[error("reduced-SWIRL wrapper prover failed: {0}")]
    Prover(String),
    #[error("reduced-SWIRL wrapper verifier failed: {0}")]
    Verifier(String),
    #[error("reduced-SWIRL wrapper public values differ")]
    PublicValues,
}

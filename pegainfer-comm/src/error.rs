//! Public error type. No wrapper-crate types appear here; backend errors
//! are erased through `Backend { source: Box<dyn Error + Send + Sync> }`.

use std::error::Error as StdError;

use thiserror::Error;

/// Result alias for `pegainfer-comm` public API.
pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for `pegainfer-comm`.
///
/// The default-feature surface uses only stable variants. Backend
/// implementations report failures through [`Error::Backend`], whose
/// `source` field carries the underlying error as a trait object so the
/// public surface stays free of wrapper-crate types.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum Error {
    /// The selected backend is not available in this build. Returned by
    /// [`crate::EpBackendBuilder::build`] when no hardware backend feature
    /// is active, or when the requested topology cannot be served.
    #[error("backend unavailable: {reason} (required feature: `{required_feature}`)")]
    BackendUnavailable {
        /// Human-readable reason.
        reason: &'static str,
        /// Cargo feature that must be enabled to construct this backend.
        required_feature: &'static str,
    },

    /// The backend code path exists (its feature is enabled) but its
    /// wiring is not yet implemented. Returned by
    /// [`crate::EpBackendBuilder::build`] while the public surface is in
    /// skeleton form, so callers cannot reach unimplemented trait
    /// bodies. Removed once the implementation lands.
    #[error("backend not yet implemented: {what}")]
    Unimplemented {
        /// Description of the unwired code path.
        what: &'static str,
    },

    /// The supplied plan is malformed or inconsistent with the backend's
    /// configured topology.
    #[error("invalid plan: {0}")]
    InvalidPlan(&'static str),

    /// The supplied buffers are malformed (size, alignment, device).
    #[error("invalid buffer: {0}")]
    InvalidBuffer(&'static str),

    /// Backend-internal failure. The underlying error is type-erased so the
    /// public surface does not depend on backend-specific types.
    #[error("backend error: {source}")]
    Backend {
        /// Erased backend error.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl Error {
    /// Wrap any `Send + Sync` error as a [`Error::Backend`]. Backend
    /// adapters use this to lift wrapper-crate errors into the public type
    /// without leaking their concrete types.
    pub fn backend<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Backend { source: Box::new(err) }
    }
}

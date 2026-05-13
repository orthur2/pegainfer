//! Operation handles + poll status.
//!
//! Each backend call returns a handle the caller drives forward via
//! [`crate::EpAllToAll::poll`]. The handles are opaque to the caller and
//! must be returned to the SAME backend that produced them.

// `from_raw` / `raw` and the inner u64 fields are intentional
// scaffolding for the upcoming backend wiring; allow dead-code while
// the public trait surface is in skeleton form.
#![allow(dead_code)]

use std::fmt;

/// Opaque dispatch handle.
///
/// Returned by [`crate::EpAllToAll::dispatch`]. The caller passes it to
/// [`crate::EpAllToAll::poll`] (wrapped as [`AnyHandle::Dispatch`]) until
/// the dispatch reports `Poll::Ready`.
#[derive(Debug)]
pub struct DispatchHandle(pub(crate) u64);

impl DispatchHandle {
    /// Construct a handle from a backend-internal id. Only the backend
    /// that produced the id should call this.
    pub(crate) fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Raw backend id. Only the backend that minted it should rely on
    /// this value.
    pub(crate) fn raw(&self) -> u64 {
        self.0
    }
}

/// Opaque combine handle. See [`DispatchHandle`].
#[derive(Debug)]
pub struct CombineHandle(pub(crate) u64);

impl CombineHandle {
    /// Construct a handle from a backend-internal id. Only the backend
    /// that produced the id should call this.
    pub(crate) fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Raw backend id. Only the backend that minted it should rely on
    /// this value.
    pub(crate) fn raw(&self) -> u64 {
        self.0
    }
}

/// Owned handle of unknown kind passed to [`crate::EpAllToAll::poll`].
#[derive(Debug)]
pub enum AnyHandle {
    /// Dispatch operation.
    Dispatch(DispatchHandle),
    /// Combine operation.
    Combine(CombineHandle),
}

/// Poll result for in-flight operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Poll {
    /// The operation completed successfully.
    Ready,
    /// The operation is still in flight; poll again later.
    Pending,
}

impl fmt::Display for Poll {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Poll::Ready => f.write_str("Ready"),
            Poll::Pending => f.write_str("Pending"),
        }
    }
}

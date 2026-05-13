//! Hardware backend implementations.
//!
//! This module and everything under it is feature-gated. The
//! default-feature build of `pegainfer-comm` never compiles any file in
//! this tree, so the public surface stays free of wrapper-crate types.
//!
//! Backends MUST NOT be re-exported through the crate root; the only way
//! to obtain one is via [`crate::EpBackendBuilder::build`].

pub(crate) mod rdma;

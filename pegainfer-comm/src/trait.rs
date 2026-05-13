//! The narrow public trait PegaInfer uses to drive an EP all-to-all
//! backend.
//!
//! The trait surface is intentionally tight: dispatch / combine / poll /
//! release, with all per-call data flowing through opaque
//! [`crate::DispatchPlan`] / [`crate::CombinePlan`] / [`crate::SendBuf`] /
//! [`crate::RecvBuf`] descriptors. No wrapper-crate type appears anywhere
//! in this signature; backend errors are erased through
//! [`crate::Error::Backend`].
//!
//! Object safety is required: PegaInfer holds the active backend as
//! `Box<dyn EpAllToAll>` inside an [`crate::EpBackend`] wrapper. All
//! methods take `&self` so backends can be shared across threads — the
//! implementation is responsible for its own internal synchronization.

use crate::buffer::{RecvBuf, SendBuf};
use crate::error::Result;
use crate::handle::{AnyHandle, CombineHandle, DispatchHandle, Poll};
use crate::plan::{CombinePlan, DispatchPlan};

/// Backend-agnostic EP all-to-all interface.
///
/// Implementations are expected to be reusable across many dispatch /
/// combine pairs once constructed via [`crate::EpBackendBuilder`].
///
/// # Concurrency
///
/// Methods take `&self`; backends must serialize their own internal
/// state. The trait requires `Send + Sync` so PegaInfer can hold the
/// backend behind `Arc` across worker threads.
///
/// # Lifetime of buffers
///
/// `send_buf` / `recv_buf` are borrowed for the duration of the call.
/// Submitting an asynchronous operation transfers logical ownership of
/// the buffer contents to the backend until the returned handle reports
/// [`Poll::Ready`]; the caller MUST keep the underlying allocations
/// alive and untouched until then.
pub trait EpAllToAll: Send + Sync {
    /// Submit a dispatch (token scatter) operation.
    ///
    /// Returns a handle the caller drives via [`Self::poll`] until it
    /// reports [`Poll::Ready`], then [`Self::release`]s.
    fn dispatch(
        &self,
        plan: &DispatchPlan,
        send_buf: &SendBuf<'_>,
        recv_buf: &mut RecvBuf<'_>,
    ) -> Result<DispatchHandle>;

    /// Submit a combine (token gather) operation paired with a prior
    /// dispatch.
    fn combine(
        &self,
        plan: &CombinePlan,
        send_buf: &SendBuf<'_>,
        recv_buf: &mut RecvBuf<'_>,
    ) -> Result<CombineHandle>;

    /// Non-blocking progress check.
    ///
    /// Returns [`Poll::Ready`] once the operation has completed and its
    /// buffers may be reused, otherwise [`Poll::Pending`].
    fn poll(&self, handle: &AnyHandle) -> Result<Poll>;

    /// Release backend resources associated with a completed handle.
    ///
    /// Must be called exactly once per handle, after [`Self::poll`]
    /// reported [`Poll::Ready`]. Calling on a still-pending handle is a
    /// programming error and may return [`crate::Error::InvalidPlan`].
    fn release(&self, handle: AnyHandle) -> Result<()>;
}

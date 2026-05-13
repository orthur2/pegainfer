//! Send / recv buffer descriptors.
//!
//! The public surface here is intentionally opaque: a buffer is a raw
//! device pointer + element count + element-size + an optional scale
//! sub-buffer descriptor. We deliberately do NOT take ownership of these
//! buffers — PegaInfer's scheduler owns the underlying allocations and
//! hands references to the backend per call.
//!
//! All `unsafe` is concentrated in the constructor: the caller asserts
//! that the pointer is valid for the lifetime of the buffer reference.

use std::marker::PhantomData;

/// Send buffer (read-only view over a device allocation).
#[derive(Debug)]
pub struct SendBuf<'a> {
    /// Raw device pointer to the token data.
    data_ptr: *const u8,
    /// Number of elements.
    num_elems: usize,
    /// Element size in bytes.
    elem_size: usize,
    /// Optional pointer to a parallel scale buffer (FP8 path).
    scale_ptr: Option<*const u8>,
    /// Lifetime marker — the buffer reference borrows the underlying
    /// allocation.
    _marker: PhantomData<&'a ()>,
}

impl<'a> SendBuf<'a> {
    /// Construct a send buffer view.
    ///
    /// # Safety
    ///
    /// `data_ptr` must point to `num_elems * elem_size` bytes of valid
    /// device memory readable from the active CUDA stream for at least
    /// `'a`. If `scale_ptr` is `Some`, it must point to a valid device
    /// allocation that pairs with `data_ptr` per the chosen backend's
    /// scale layout.
    pub unsafe fn new(
        data_ptr: *const u8,
        num_elems: usize,
        elem_size: usize,
        scale_ptr: Option<*const u8>,
    ) -> Self {
        Self { data_ptr, num_elems, elem_size, scale_ptr, _marker: PhantomData }
    }

    /// Raw device pointer.
    pub fn data_ptr(&self) -> *const u8 {
        self.data_ptr
    }

    /// Number of elements.
    pub fn num_elems(&self) -> usize {
        self.num_elems
    }

    /// Element size in bytes.
    pub fn elem_size(&self) -> usize {
        self.elem_size
    }

    /// Optional FP8-scale companion pointer.
    pub fn scale_ptr(&self) -> Option<*const u8> {
        self.scale_ptr
    }
}

/// Recv buffer (writable view over a device allocation).
#[derive(Debug)]
pub struct RecvBuf<'a> {
    /// Raw device pointer to the destination.
    data_ptr: *mut u8,
    /// Capacity in elements.
    capacity: usize,
    /// Element size in bytes.
    elem_size: usize,
    /// Lifetime marker.
    _marker: PhantomData<&'a mut ()>,
}

impl<'a> RecvBuf<'a> {
    /// Construct a recv buffer view.
    ///
    /// # Safety
    ///
    /// `data_ptr` must point to `capacity * elem_size` bytes of valid,
    /// uniquely-owned device memory writable from the active CUDA stream
    /// for at least `'a`.
    pub unsafe fn new(data_ptr: *mut u8, capacity: usize, elem_size: usize) -> Self {
        Self { data_ptr, capacity, elem_size, _marker: PhantomData }
    }

    /// Raw device pointer.
    pub fn data_ptr(&self) -> *mut u8 {
        self.data_ptr
    }

    /// Capacity in elements.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Element size in bytes.
    pub fn elem_size(&self) -> usize {
        self.elem_size
    }
}

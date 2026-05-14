#![cfg(feature = "deepseek-v4")]

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;

use anyhow::{Result, ensure};
use cudarc::driver::sys::{CUresult, CUstream};
use half::bf16;
use pegainfer_kernels::ffi;

const HEAD_DIM: usize = 128;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy + Default> DeviceBuffer<T> {
    fn from_host(data: &[T]) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        let bytes = data.len() * size_of::<T>();
        cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) })?;
        if bytes > 0 {
            cuda_check(unsafe {
                cudaMemcpy(
                    ptr,
                    data.as_ptr().cast::<c_void>(),
                    bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            })?;
        }
        Ok(Self {
            ptr: ptr.cast::<T>(),
            len: data.len(),
        })
    }

    fn copy_to_host(&self) -> Result<Vec<T>> {
        let mut data = vec![T::default(); self.len];
        let bytes = self.len * size_of::<T>();
        if bytes > 0 {
            cuda_check(unsafe {
                cudaMemcpy(
                    data.as_mut_ptr().cast::<c_void>(),
                    self.ptr.cast::<c_void>(),
                    bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            })?;
        }
        Ok(data)
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr.cast::<c_void>());
            }
        }
    }
}

fn cuda_check(code: i32) -> Result<()> {
    ensure!(code == 0, "CUDA runtime call failed with code {code}");
    Ok(())
}

fn assert_cuda_success(result: CUresult) {
    assert_eq!(result, CUresult::CUDA_SUCCESS);
}

fn patterned_bf16(len: usize, scale: f32, bias: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let v = ((i * 37 + 11) % 29) as f32 - 14.0;
            bf16::from_f32(v * scale + bias)
        })
        .collect()
}

fn bf16_words(values: &[bf16]) -> Vec<ffi::Half> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn reference_prefill(
    q: &[bf16],
    kv: &[bf16],
    weights: &[bf16],
    seq_len: usize,
    local_heads: usize,
    compressed_len: usize,
    score_scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; seq_len * compressed_len];
    for token in 0..seq_len {
        for compressed in 0..compressed_len {
            let mut acc = 0.0f32;
            for head in 0..local_heads {
                let row = token * local_heads + head;
                let mut dot = 0.0f32;
                for k in 0..HEAD_DIM {
                    dot += q[row * HEAD_DIM + k].to_f32() * kv[compressed * HEAD_DIM + k].to_f32();
                }
                acc += dot.max(0.0) * weights[row].to_f32();
            }
            scores[token * compressed_len + compressed] = acc * score_scale;
        }
    }
    scores
}

fn reference_decode(
    q: &[bf16],
    kv: &[bf16],
    weights: &[bf16],
    local_heads: usize,
    compressed_len: usize,
    score_scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; compressed_len];
    for compressed in 0..compressed_len {
        let mut acc = 0.0f32;
        for head in 0..local_heads {
            let mut dot = 0.0f32;
            for k in 0..HEAD_DIM {
                dot += q[head * HEAD_DIM + k].to_f32() * kv[compressed * HEAD_DIM + k].to_f32();
            }
            acc += dot.max(0.0) * weights[head].to_f32();
        }
        scores[compressed] = acc * score_scale;
    }
    scores
}

fn assert_close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let diff = (got - expected).abs();
        let tol = 2.5e-3f32.max(expected.abs() * 2.5e-2);
        assert!(
            diff <= tol,
            "mismatch at {idx}: got={got}, expected={expected}, diff={diff}, tol={tol}"
        );
    }
}

#[test]
fn indexer_scores_prefill_cutedsl_aot_matches_reference() -> Result<()> {
    let seq_len = 4usize;
    let local_heads = 8usize;
    let compressed_len = 16usize;
    let score_scale = 0.125f32;
    let rows = seq_len * local_heads;

    let q = patterned_bf16(rows * HEAD_DIM, 0.003, 0.001);
    let kv = patterned_bf16(compressed_len * HEAD_DIM, -0.002, 0.002);
    let weights = patterned_bf16(rows, 0.01, 0.05);
    let expected = reference_prefill(
        &q,
        &kv,
        &weights,
        seq_len,
        local_heads,
        compressed_len,
        score_scale,
    );

    let q_d = DeviceBuffer::from_host(&bf16_words(&q))?;
    let kv_d = DeviceBuffer::from_host(&bf16_words(&kv))?;
    let weights_d = DeviceBuffer::from_host(&bf16_words(&weights))?;
    let mut scores_d = DeviceBuffer::from_host(&vec![0.0f32; expected.len()])?;
    let stream: CUstream = ptr::null_mut();

    let result = unsafe {
        ffi::deepseek_indexer_scores_prefill_cuda(
            q_d.as_ptr(),
            kv_d.as_ptr(),
            weights_d.as_ptr(),
            scores_d.as_mut_ptr(),
            seq_len as i32,
            local_heads as i32,
            HEAD_DIM as i32,
            compressed_len as i32,
            score_scale,
            stream,
        )
    };
    assert_cuda_success(result);
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let got = scores_d.copy_to_host()?;
    assert_close(&got, &expected);
    Ok(())
}

#[test]
fn indexer_scores_decode_cutedsl_aot_matches_reference() -> Result<()> {
    let local_heads = 8usize;
    let compressed_len = 8usize;
    let score_scale = 0.25f32;

    let q = patterned_bf16(local_heads * HEAD_DIM, 0.002, -0.001);
    let kv = patterned_bf16(compressed_len * HEAD_DIM, 0.003, 0.001);
    let weights = patterned_bf16(local_heads, 0.02, 0.08);
    let expected = reference_decode(&q, &kv, &weights, local_heads, compressed_len, score_scale);

    let q_d = DeviceBuffer::from_host(&bf16_words(&q))?;
    let kv_d = DeviceBuffer::from_host(&bf16_words(&kv))?;
    let weights_d = DeviceBuffer::from_host(&bf16_words(&weights))?;
    let mut scores_d = DeviceBuffer::from_host(&vec![0.0f32; expected.len()])?;
    let stream: CUstream = ptr::null_mut();

    let result = unsafe {
        ffi::deepseek_indexer_scores_decode_cuda(
            q_d.as_ptr(),
            kv_d.as_ptr(),
            weights_d.as_ptr(),
            scores_d.as_mut_ptr(),
            local_heads as i32,
            HEAD_DIM as i32,
            compressed_len as i32,
            score_scale,
            stream,
        )
    };
    assert_cuda_success(result);
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let got = scores_d.copy_to_host()?;
    assert_close(&got, &expected);
    Ok(())
}

#[test]
fn indexer_scores_decode_cutedsl_aot_handles_single_compressed_block() -> Result<()> {
    let local_heads = 8usize;
    let compressed_len = 1usize;
    let score_scale = 0.25f32;

    let q = patterned_bf16(local_heads * HEAD_DIM, 0.002, -0.001);
    let kv = patterned_bf16(compressed_len * HEAD_DIM, 0.003, 0.001);
    let weights = patterned_bf16(local_heads, 0.02, 0.08);
    let expected = reference_decode(&q, &kv, &weights, local_heads, compressed_len, score_scale);

    let q_d = DeviceBuffer::from_host(&bf16_words(&q))?;
    let kv_d = DeviceBuffer::from_host(&bf16_words(&kv))?;
    let weights_d = DeviceBuffer::from_host(&bf16_words(&weights))?;
    let mut scores_d = DeviceBuffer::from_host(&vec![0.0f32; expected.len()])?;
    let stream: CUstream = ptr::null_mut();

    let result = unsafe {
        ffi::deepseek_indexer_scores_decode_cuda(
            q_d.as_ptr(),
            kv_d.as_ptr(),
            weights_d.as_ptr(),
            scores_d.as_mut_ptr(),
            local_heads as i32,
            HEAD_DIM as i32,
            compressed_len as i32,
            score_scale,
            stream,
        )
    };
    assert_cuda_success(result);
    cuda_check(unsafe { cudaDeviceSynchronize() })?;

    let got = scores_d.copy_to_host()?;
    assert_close(&got, &expected);
    Ok(())
}

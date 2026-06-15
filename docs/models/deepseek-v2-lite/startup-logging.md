# DeepSeek-V2-Lite Startup Logging

> **TL;DR:** DeepSeek-V2-Lite EP2 now emits startup/load logs for config, manifest validation, rank weight loads, backend init, and engine readiness; local compile verification is blocked by a missing FlashInfer include tree.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - routed the task to the DeepSeek-V2-Lite model docs and Kimi-K2 reference docs.
  - `docs/models/deepseek-v2-lite/status.md` - confirmed DeepSeek-V2-Lite is a correctness and diagnostic EP2 target; avoid broad serving or graph-readiness claims.
  - `docs/models/deepseek-v2-lite/source-layout.md` - confirmed `runtime/generation.rs` owns load/orchestration and `runtime/backend.rs` owns backend parsing/runtime setup.
  - `docs/models/kimi-k2/source-layout.md` - confirmed Kimi startup/load logic is split into runner bring-up and worker responsibilities.
  - `docs/models/kimi-k2/roadmap.md` - confirmed Kimi's current observability is tied to startup/runtime milestones, not a generic logging framework.
  - `openinfer-kimi-k2/src/runner/bringup.rs` - found the reference pattern: `info!` phase starts/completions with elapsed seconds and `debug!` per-rank/load-plan details.
  - `openinfer-deepseek-v2-lite/src/runtime/generation.rs` - found the DeepSeek-V2-Lite load path and phase boundaries.
  - `openinfer-deepseek-v2-lite/src/runtime/backend.rs` - found env-driven EP backend parsing and runtime initialization.
  - `openinfer-deepseek-v2-lite/src/engine.rs` - found the server engine wrapper around the generator.
  - `openinfer-deepseek-v2-lite/src/weights.rs` - found manifest and rank plan metadata that can support debug counts.
  - `openinfer-deepseek-v2-lite/src/model.rs` - found rank-local model loading boundaries.
  - `openinfer-deepseek-v2-lite/Cargo.toml` - confirmed the crate already depends on workspace `log`.
- **Relevant history**:
  - `docs/models/deepseek-v2-lite/source-layout.md` - the source split preserved exact HF / host-staged / NCCL gates; this logging task should stay behavior-preserving.
  - `docs/models/deepseek-v2-lite/status.md` - host-staged and NCCL are correctness/diagnostic backends; logging must not imply production serving parity or graph readiness.
- **Plan**:
  1. Add `log::{debug, info}` and `Instant`-timed phase logs to `openinfer-deepseek-v2-lite/src/runtime/generation.rs` around config validation, manifest/rank-plan validation, rank 0 load, rank 1 load, and EP backend initialization.
  2. Add small metadata accessors in `openinfer-deepseek-v2-lite/src/weights.rs` if needed so debug logs can report manifest and per-rank tensor counts without exposing internals broadly.
  3. Add an engine-level startup/ready log in `openinfer-deepseek-v2-lite/src/engine.rs` only if it complements, rather than duplicates, the generator load logs.
  4. Run `cargo fmt --all --check` and a focused compile check for `openinfer-deepseek-v2-lite` with the `deepseek-v2-lite` feature.
- **Risks / open questions**:
  - The focused compile may require CUDA toolchain availability; if the local environment cannot compile it, record the exact failure and keep verification to formatting plus the compile attempt.

## Execution Log

### Step 1: Add startup/load phase logging

- Updated `openinfer-deepseek-v2-lite/src/runtime/generation.rs`.
- Added `info!` milestones mirroring Kimi-K2's bring-up style:
  - startup resolution with model path, devices, and CUDA graph flag;
  - config load/validation timing and model shape summary;
  - backend/device validation;
  - weight manifest validation timing;
  - rank 0 and rank 1 weight load start/completion timing;
  - EP backend initialization timing;
  - total generator load timing.
- Added `debug!` load-plan details for rank-owned expert ranges.
- Result: code changed, verification pending.

### Step 2: Add small metadata accessors

- Updated `openinfer-deepseek-v2-lite/src/weights.rs`.
- Added `ModelManifest::tensor_count()` and `RankLoadPlan::tensor_count()` so logs can report useful manifest/plan counts without exposing tensor internals.
- Result: code changed, verification pending.

### Step 3: Add engine ready log

- Updated `openinfer-deepseek-v2-lite/src/engine.rs`.
- Added an engine-level start/ready log around generator load and engine thread spawn.
- Result: code changed, verification pending.

### Step 4: Verify

- Ran `cargo fmt --all --check`.
- Result: passed.
- Ran `git diff --check`.
- Result: passed.
- Ran `cargo check --release -p openinfer-deepseek-v2-lite --features deepseek-v2-lite --lib --tests`.
- Result: failed before checking `openinfer-deepseek-v2-lite` because `openinfer-kernels` could not compile FlashInfer-backed CUDA translation units:

```text
fatal error: flashinfer/attention/decode.cuh: No such file or directory
fatal error: flashinfer/sampling.cuh: No such file or directory
fatal error: flashinfer/norm.cuh: No such file or directory
```

- Checked local FlashInfer include candidates used by `openinfer-kernels/build.rs`.
- Result: no `flashinfer/sampling.cuh` include tree was present; `openinfer-kernels/third_party/flashinfer` exists but is empty.

## Debrief

- **Outcome**: DeepSeek-V2-Lite EP2 startup now mirrors the Kimi-K2 logging style with `info!` phase milestones and `debug!` load-plan detail. No generation, backend, scheduling, or kernel behavior was changed.
- **Pitfalls encountered**:
  - The focused compile gate is environment-blocked by missing FlashInfer headers in `openinfer-kernels/third_party/flashinfer` and no fallback include tree under `.venv`.
  - The workspace already had a pending `log` dependency addition for `openinfer-deepseek-v2-lite`; the new imports rely on that dependency.
- **Lessons learned**:
  - DeepSeek-V2-Lite has clean logging boundaries at generator load time: config, manifest/plan validation, rank0 load, rank1 load, backend init, and engine ready.
- **Follow-ups**:
  - Populate FlashInfer headers or set `OPENINFER_FLASHINFER_INCLUDE`, then rerun the focused `cargo check`.

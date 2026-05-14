# DeepSeek V4 HTTP Serving Benchmark Gate

**Created**: 2026-05-14
**Status**: active
**Canonical task**: task #18; P1 TTFT/sweep extension in `#Dsv4性能调优` task #4

## Purpose

This gate measures the OpenAI-compatible HTTP endpoint under concurrent load. It
does not use the in-process `bench_serving request` path as serving evidence.

The benchmark client sends streaming `/v1/completions` requests and records:

- QPS and completed/failed/timeout counts.
- TTFT from request send to first streamed text chunk.
- ITL and TPOT from streamed text chunks.
- End-to-end latency percentiles.
- Per-request output hashes plus a combined output hash for reproducibility.

The P1 sweep extension adds server-log trace attribution for successful
requests:

- `frontend_to_queue_ms`: client request start to engine queue timestamp. This
  includes HTTP ingress, tokenization, and vLLM request submission.
- `admission_queue_ms`: engine queued to scheduled.
- `prefill_ms`: DSV4 direct prefill plus decode-cache seeding.
- `first_decode_ms`: first decode step after the first streamed token. In the
  current direct path, the first streamed token is sampled from prefill logits,
  so this phase explains early TPOT rather than TTFT.
- `stream_flush_ms`: server first-token emission to client first text chunk.

## Reproducible Commands

Build the server on the target GPU host:

```bash
cd /path/to/pegainfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.1
export PEGAINFER_TILELANG_PYTHON=/path/to/venv/bin/python
export PEGAINFER_TRITON_PYTHON=/path/to/venv/bin/python
export PEGAINFER_NVCC_JOBS=8
export CARGO_TARGET_DIR=/path/to/pegainfer-target

cargo build --release -p pegainfer-server --features deepseek-v4 --bin pegainfer
```

Start the OpenAI-compatible HTTP endpoint:

```bash
$CARGO_TARGET_DIR/release/pegainfer \
  --model-path /data/DeepSeek-V4-Flash \
  --port 18118 2>&1 | tee /tmp/dsv4_http_server.log
```

Verify the model endpoint:

```bash
curl -sS http://127.0.0.1:18118/v1/models
```

Run the HTTP serving benchmark:

```bash
python3 scripts/bench_http_serving.py \
  --base-url http://127.0.0.1:18118 \
  --model /data/DeepSeek-V4-Flash \
  --warmup 2 \
  --num-requests 8 \
  --concurrency 2 \
  --prompt-words 16 \
  --max-tokens 16 \
  --timeout 240 \
  --server-log /tmp/dsv4_http_server.log \
  --out /tmp/dsv4_http_bench_task18.json
```

Run the P1 concurrency / max-token sweep:

```bash
python3 scripts/bench_http_sweep.py \
  --base-url http://127.0.0.1:18118 \
  --model /data/DeepSeek-V4-Flash \
  --warmup 2 \
  --num-requests 8 \
  --concurrency 1,2,4,8 \
  --max-tokens 16 \
  --repeats 3 \
  --prompt-words 16 \
  --timeout 240 \
  --server-log /tmp/dsv4_http_server.log \
  --out-dir /tmp/dsv4_http_sweep_task4
```

The script is intentionally model-server agnostic at the HTTP layer. It only
requires an OpenAI-compatible `/v1/completions` endpoint that supports streaming
responses.

The server trace columns are pegainfer-specific and require a pegainfer server
log containing `pegainfer_http_trace` lines. The sweep fails when any cell has
request failures/timeouts or per-request output hashes that change across
repeats.

## Current Evidence

Evidence below was collected on the internal 8-GPU DeepSeek-V4-Flash validation
host. It describes only this commit, machine, endpoint, and harness.

| Field | Value |
| --- | --- |
| Commit | PR body records the validated head; tracked docs avoid self-referential commit hashes. |
| Endpoint | OpenAI-compatible `/v1/completions`, streaming |
| Model | `/data/DeepSeek-V4-Flash` |
| Workload | warmup `2`, measured requests `8`, concurrency `2`, prompt words `16`, max tokens `16`, temperature `0`, ignore EOS `true`, timeout `240s` |
| Result | completed `8`, failed `0`, timeout `0`, error rate `0.0` |
| QPS | `1.6869` completed requests/s |
| Latency | avg `1112.19ms`, p50 `1179.70ms`, p95 `1207.23ms`, p99 `1207.23ms` |
| TTFT | avg `680.38ms`, p50 `746.66ms`, p95 `775.06ms`, p99 `775.06ms` |
| TPOT | avg `28.78ms`, p50 `28.81ms`, p95 `28.88ms`, p99 `28.88ms` |
| ITL | avg `28.78ms`, p50 `28.28ms`, p95 `30.57ms`, p99 `30.74ms` |
| Output stability | output chunks `128`, unique output hashes `8`, combined output hash `22706877075acde0` |

### P1 TTFT Trace And Concurrency Sweep

The P1 sweep was collected after the HTTP correctness gate stabilized. It keeps
the same prompt shape and `max_tokens=16`, repeats each concurrency point three
times, and treats per-request hash drift as a correctness failure before any
performance interpretation.

| Concurrency | Correctness | QPS (3 runs) | TTFT avg ms (3 runs) | TPOT avg ms (3 runs) | Dominant trace attribution |
| --- | --- | --- | --- | --- | --- |
| `1` | pass; failed `0`, timeout `0`, combined hash `22706877075acde0` in all runs | `1.649`, `1.638`, `1.642` | `165.9`, `167.3`, `168.8` | `29.36`, `29.52`, `29.34` | admission queue avg `0.0ms`; prefill avg `~165ms` |
| `2` | pass; failed `0`, timeout `0`, combined hash `22706877075acde0` in all runs | `1.650`, `1.649`, `1.651` | `696.8`, `697.6`, `695.9` | `29.34`, `29.34`, `29.32` | admission queue avg `~530ms`; prefill avg `~166ms` |
| `4` | pass; failed `0`, timeout `0`, combined hash `22706877075acde0` in all runs | `1.649`, `1.646`, `1.651` | `1531.3`, `1536.2`, `1530.2` | `29.35`, `29.34`, `29.30` | admission queue avg `~1365ms`; prefill avg `~166ms` |
| `8` | pass; failed `0`, timeout `0`, combined hash `22706877075acde0` in all runs | `1.649`, `1.648`, `1.652` | `2291.7`, `2290.4`, `2285.0` | `29.29`, `29.35`, `29.30` | admission queue avg `~2124ms`; prefill avg `~167ms` |

Trace attribution shows that the c1 to c8 TTFT increase is dominated by
admission queue time under the current HTTP serving path, while `prefill_ms`,
`first_decode_ms`, `stream_flush_ms`, and TPOT remain roughly stable. This is
diagnostic evidence for the current single-request scheduler turn used by HTTP
serving; it is not a throughput optimization result.

## Boundary

This PR establishes a benchmark gate and one real HTTP run. It does not claim
vLLM parity, production serving stability, larger batch scalability, paged or
prefix KV, or P/D handoff behavior.

`bench_serving request` remains the in-process direct regression path. It is not
used as a substitute for HTTP serving metrics in this document.

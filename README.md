# LongCache — On-Device Long-Context LLM via KV-Cache Compression

Holds long conversations on a fixed memory budget by compressing the LLM's key-value cache, on Apple Silicon with MLX. Includes a learned (contextual-bandit) eviction policy benchmarked head-to-head against the standard heuristics.

## The problem

On-device LLMs choke on long conversations because of the KV cache. Every token added to the context permanently adds to a key-value cache the model must hold in RAM, and that cache grows linearly with the conversation. On a device with fixed unified memory, a long chat slows generation and eventually crashes with an out-of-memory error — and offloading to the cloud destroys the privacy that is the entire reason to run on-device. The problem: hold the longest possible conversation, generating as fast as possible, on a fixed memory budget, while losing as little model quality as possible. The hard part is the tradeoff — evict old tokens and the model forgets (quality drops); keep everything and it crashes.

## Approach

Establish a baseline, then attack the cache with three levers, measuring every step against the baseline on a small model that fits the hardware:
1. **Quantize what you keep** — store cached K/V in INT4/INT8 instead of FP16.
2. **Keep fewer tokens** — evict with proven heuristics (recency window + heavy-hitter, the StreamingLLM/H2O family).
3. **Learn the eviction policy** — frame keep/evict as a contextual bandit and benchmark it against the heuristics.

Quality is measured two ways, both offline: perplexity on held-out long text, and needle-in-a-haystack retrieval accuracy vs context length.

## Results

Measured on Apple Silicon, not estimated. Filled in as each phase completes.

> **Measurement status (2026-06-06).** Phase 0 code is complete, compiles, and is built
> against the verified current mlx-lm API. It has **not** been run, because the current
> build host is an Intel Mac (Intel Iris Plus GPU, Python 3.9) and MLX is Apple-Silicon-only
> — there is no Intel-Mac MLX backend or wheel. Per the project's cardinal rule, every metric
> below is **TBD** until `scripts/baseline.py` is run on an M-series Mac (macOS 14+, Python
> 3.10+). The script refuses to run on unsupported hardware rather than emit a fabricated
> number. No value here is estimated or extrapolated.

### Baseline (Phase 0)

| Context (tokens) | TTFT (s) | Decode tok/s | Peak memory (GB) | KV memory (GB) | Perplexity | Needle acc. |
|---|---|---|---|---|---|---|
| 2k | TBD | TBD | TBD | TBD | TBD | TBD |
| 4k | TBD | TBD | TBD | TBD | TBD | TBD |
| 8k | TBD | TBD | TBD | TBD | TBD | TBD |
| 16k | TBD | TBD | TBD | TBD | TBD | TBD |
| OOM ceiling | — | — | — | — | — | — |

KV memory is measured directly from the live cache arrays (`sum(nbytes)`); peak memory from
`mx.get_peak_memory()`. The analytical KV size `2 · layers · kv_heads · head_dim · seq · 2B`
(FP16) is also computed at load time as a fit check. OOM ceiling = largest context whose
measured peak stays under 80% of unified memory.

### KV quantization (Phase 1)

All tokens are kept; each costs less by storing cached K/V quantized (group size 64) via
mlx-lm's `QuantizedKVCache`. KV memory is the **measured** compressed cache size
(packed uint32 + per-group fp16 scales/biases), not an analytical estimate. Perplexity is
computed by a teacher-forced forward pass that runs *through the quantized cache*, so the
quality delta reflects real quantized-attention error, not an uncached approximation. Headline
row is 16k context; `scripts/quantize.py` emits the full 2k/4k/8k/16k sweep.

| Context | Precision | KV mem (GB) | KV vs FP16 | Peak mem (GB) | Perplexity | PPL Δ | Needle acc. | Decode tok/s |
|---|---|---|---|---|---|---|---|---|
| 16k | fp16 | TBD | 1.00x | TBD | TBD | 0.00 (ref) | TBD | TBD |
| 16k | int8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 16k | int4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Status as Phase 0: code complete, API-verified, compiles; numbers TBD pending an Apple
Silicon run (this build host is an Intel Mac that cannot run MLX).

### Heuristic eviction (Phase 2)

Three budget-limited strategies keep the same number of tokens (default budget 2048 at 16k
context, an 8x cut) and are scored against the full keep-everything cache:

- **recency** — `RotatingKVCache(keep=0)`, a pure sliding window.
- **streaming** — `RotatingKVCache(keep=sink)`, StreamingLLM: attention sinks + recent window.
- **heavy_hitter** — H2O: attention sinks + recent window + the highest-attention tokens.

Recency and StreamingLLM ride mlx-lm's fused attention. H2O needs per-token attention scores,
which the fused kernel never exposes, so it runs through a scope-patched explicit-softmax
attention that captures per-key mass, driven one token at a time (survivors keep absolute
RoPE; single-query steps keep the score matrix tiny and the mask trivial). That token-wise
prefill is H2O's honest runtime cost — its TTFT will be high, which the table reports rather
than hides.

| Strategy | KV mem (GB) | KV vs full | Peak mem (GB) | Perplexity | PPL Δ vs full | Needle acc. | Decode tok/s | TTFT (s) |
|---|---|---|---|---|---|---|---|---|
| full | TBD | 1.00x (ref) | TBD | TBD | 0.00 (ref) | TBD | TBD | TBD |
| recency | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| streaming | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| heavy_hitter | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

`scripts/evict.py` emits this table. Status as Phases 0–1: code complete, API-verified,
compiles; eviction index math validated in pure Python; numbers TBD pending an Apple Silicon
run (this build host is an Intel Mac that cannot run MLX).

### Learned eviction (Phase 3)

Not started — contextual-bandit policy benchmarked head-to-head against the Phase 2 heuristics.

## Architecture

```
mlx-lm small 4-bit model
  -> baseline generation loop + telemetry (TTFT, tok/s, peak + KV memory, perplexity, needle accuracy)
  -> KV-cache quantization (compress what you keep)
  -> eviction: recency + heavy-hitter heuristics (keep fewer tokens)
  -> learned contextual-bandit eviction policy, benchmarked vs the heuristics
  -> stress benchmark to 16k+ -> comparative table
```

## Phases

- **Phase 0** — Baseline long-context loop + KV-memory and quality telemetry.
- **Phase 1** — KV-cache quantization.
- **Phase 2** — Heuristic eviction (recency + heavy-hitter).
- **Phase 3** — Learned (contextual-bandit) eviction, benchmarked vs heuristics.
- **Phase 4** — Stress benchmark to 16k+ and the master comparative table.
- **Phase 5** — Docs, resume bullets, and a memory-vs-context chart.

## Setup

Requires an Apple Silicon Mac (M1/M2/M3/M4), macOS 14+, Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_data.py    # downloads held-out text (gitignored)
python scripts/baseline.py    # Phase 0 baseline sweep + telemetry
```

Default model: `mlx-community/Llama-3.2-1B-Instruct-4bit` (4-bit, ~128k context window),
chosen to fit a 16 GB machine with ample headroom so KV growth — not the weights — is the
binding constraint. `scripts/baseline.py` checks the model fits the unified-memory budget
before generating and aborts with a clear message on unsupported hardware.

## Why this maps to the role

On-device LLM inference, optimization, linear algebra, and a reinforcement-learning (contextual-bandit) policy, all on Apple Silicon, with measured numbers and an honest comparison behind every claim.

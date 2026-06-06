# Progress

## Current status
Phase 0 **code complete and verified-as-far-as-possible on this host**; **measurements TBD**.
The build machine is an Intel Mac (Intel Core i5-1038NG7, Intel Iris Plus GPU, 16 GB,
Python 3.9), on which MLX cannot run — MLX is Apple-Silicon-only and ships no Intel-Mac
wheel or backend. All Phase 0 code is written against the verified current mlx-lm API and
must be run on an Apple Silicon Mac (macOS 14+, Python 3.10+) to produce real numbers.

## Phase checklist
- [~] Phase 0 — Baseline long-context loop + KV-memory and quality telemetry (code done; numbers TBD pending Apple Silicon)
- [~] Phase 1 — KV-cache quantization (INT8 / INT4) (code done; numbers TBD pending Apple Silicon)
- [ ] Phase 2 — Heuristic eviction (recency + attention-sink + heavy-hitter)
- [ ] Phase 3 — Learned (contextual-bandit) eviction, benchmarked vs heuristics
- [ ] Phase 4 — Stress benchmark to 16k+ and master comparative table
- [ ] Phase 5 — Docs, resume bullets, memory-vs-context chart

## What Phase 0 built
- `longcache/preflight.py` — stdlib-only gate; refuses to run on non-Apple-Silicon / Python < 3.10 with the real reason.
- `longcache/config.py` — `BaselineConfig` (model id, context lengths, decode tokens, perplexity/needle/OOM settings).
- `longcache/telemetry.py` — `Stopwatch`, `MemoryProbe` (`mx.get_peak_memory`/`reset_peak_memory`, bytes), live `kv_cache_bytes`.
- `longcache/model_runtime.py` — model load, config introspection (layers/kv_heads/head_dim), measured weight bytes, analytical KV-size fit check, cache construction.
- `longcache/textdata.py` — held-out text load + tokenization helpers.
- `longcache/perplexity.py` — sliding-window perplexity (HF-style stride) via `nn.losses.cross_entropy`.
- `longcache/generation.py` — deterministic (argmax) generation via `generate_step`; measures TTFT, decode tok/s, peak + KV memory.
- `longcache/needle.py` — needle-in-a-haystack builder/scorer across insertion depth and context length.
- `longcache/benchmark.py` — orchestrates the sweep, OOM-ceiling probe, emits the markdown table + raw JSON.
- `scripts/baseline.py` — Phase 0 entrypoint (preflight → fit report → sweep → table).
- `scripts/get_data.py` — downloads public-domain held-out text (gitignored).

## What Phase 1 built
- `longcache/model_runtime.py::quantized_cache` — builds a from-empty `QuantizedKVCache` per layer via `to_quantized(group_size, bits)`.
- `longcache/generation.py` — generation now takes `kv_bits`/`kv_group_size`/`quantized_kv_start`, passed to `generate_step`; the cache is quantized in place so measured KV bytes are the real compressed size.
- `longcache/cached_perplexity.py` — teacher-forced perplexity through the live cache; with a quantized cache, attention routes through `quantized_scaled_dot_product_attention`, so the delta is the true quantized-attention quality cost (verified against upstream `models/base.py`).
- `longcache/quantization_benchmark.py` — FP16/INT8/INT4 sweep across context lengths; emits before/after table (KV mem, KV-vs-FP16 ratio, peak mem, perplexity + Δ, needle acc, decode tok/s) + raw JSON.
- `scripts/quantize.py` — Phase 1 entrypoint.
- Design note: perplexity quantizes from token 0 (most aggressive, cleanest delta); generation uses `generate_step` with `quantized_kv_start=0` (prefill fp16, decode quantized) — both legitimate, noted for the interview.

## Verified on this host
- All modules compile (`python3 -m py_compile`).
- `scripts/baseline.py` exits 2 on this Intel Mac with the correct hardware/Python diagnosis.
- mlx-lm API names checked against current upstream source (load, generate_step, make_prompt_cache, KVCache/QuantizedKVCache/RotatingKVCache, mx memory fns).

## Known issues / risks
- **Cannot measure anything here** — needs Apple Silicon. This is the single blocker to filling every table.
- `model.args` field names are introspected defensively; verify against the chosen model on first real run.
- Needle prompt length ≈ target context + question + chat-template overhead; harness records the actual measured prompt length.
- Held-out text is public-domain (may be in pretraining); perplexity is a relative baseline for measuring deltas, not unseen-data generalization.

## Next action
On an Apple Silicon Mac: `pip install -r requirements.txt`, `python scripts/get_data.py`,
then `python scripts/baseline.py` (Phase 0) and `python scripts/quantize.py` (Phase 1).
Paste the printed tables back; I fill README / RESUME_BULLETS / this file with the real
numbers (baseline + OOM ceiling + FP16/INT8/INT4 deltas), then proceed to Phase 2
(heuristic eviction: recency window, attention-sink + recent, heavy-hitter).

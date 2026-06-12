# Progress

## Current status
**Phases 0–5 code complete and verified-as-far-as-possible on this host**; **measurements TBD**.
The build machine is an Intel Mac (Intel Core i5-1038NG7, Intel Iris Plus GPU, 16 GB,
Python 3.9), on which MLX cannot run — MLX is Apple-Silicon-only and ships no Intel-Mac
wheel or backend. All code is written against the verified current mlx-lm API and must be run
on an Apple Silicon Mac (macOS 14+, Python 3.10+) to produce real numbers and charts.

## Phase checklist
- [~] Phase 0 — Baseline long-context loop + KV-memory and quality telemetry (code done; numbers TBD pending Apple Silicon)
- [~] Phase 1 — KV-cache quantization (INT8 / INT4) (code done; numbers TBD pending Apple Silicon)
- [~] Phase 2 — Heuristic eviction (recency + attention-sink + heavy-hitter) (code done; numbers TBD pending Apple Silicon)
- [~] Phase 3 — Learned (contextual-bandit) eviction, benchmarked vs heuristics (code done; numbers TBD pending Apple Silicon)
- [~] Phase 4 — Stress benchmark to 16k+ and master comparative table (code done; numbers TBD pending Apple Silicon)
- [~] Phase 5 — Docs, resume bullets, memory-vs-context chart (code + docs done; charts render once Phase 4 data exists)

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

## What Phase 2 built
- `longcache/model_runtime.py::rotating_cache` — per-layer `RotatingKVCache(max_size, keep)`; `keep=0` = recency window, `keep=sink` = StreamingLLM.
- `longcache/heavy_hitter.py` — H2O: `HeavyHitterCache` (sink + recent + highest-attention, fixed budget), a scope-patched explicit-softmax `scaled_dot_product_attention` that captures per-key attention mass (the fused kernel never exposes weights), and `HeavyHitterRunner` (token-by-token prefill/decode + perplexity). Verified against upstream: model calls the attention helper as a bare module name (patchable), RoPE applied before `update_and_fetch` (survivors keep absolute positions), `create_attention_mask` returns None at N==1 (token-wise steps need no mask).
- `longcache/eviction_benchmark.py` + `scripts/evict.py` — compares full / recency / streaming / heavy_hitter at 16k on KV mem, KV-vs-full ratio, peak mem, perplexity + Δ, needle acc, decode tok/s, TTFT.
- `longcache/needle.py` — `run_needle_suite` now takes a `cache_factory` so each needle depth gets a fresh strategy cache.
- Validation done here: eviction index math checked in pure Python (sink + recent always retained, middle keeps top-score heavy hitters, temporal order preserved, size capped at budget).
- Known cost: H2O prefill is un-fused and token-by-token (TTFT will be high) — the honest price of scoring attention the fused kernel hides; recency/streaming use the fast fused path. H2O TTFT is reported, not hidden.

## What Phase 3 built
- `longcache/heavy_hitter.py` — refactored the H2O runner into a generic `TokenStreamRunner` (cache-factory based); attention capture now duck-types on `record_scores` so any eviction cache can receive scores. `HeavyHitterRunner` keeps its signature (eviction_benchmark unchanged).
- `longcache/learned_eviction.py` — the contextual bandit: `RolloutCache` + `RolloutCollector` (one token-by-token pass over a full cache under capture; labels each token with future attention over a window), `collect_rollouts`, `train_policy` (sklearn `StandardScaler` + `Ridge`, plus a past-attention-only baseline R² so we can see if the extra features add signal), `LinearPolicy` (JSON-serializable), `LearnedCache` (H2O-style budget eviction but the eviction score is the learned linear prediction from `[avg attention, position fraction, value norm, layer]`, scored with cheap mx ops), and `LearnedRunner`.
- `longcache/eviction_benchmark.py` — `_measure_token_stream` generalized; `run(learned_policy=...)` appends the learned row.
- `longcache/learned_benchmark.py` + `scripts/learn.py` — collect → train → benchmark all 5 strategies → markdown table + explicit verdict (compares learned vs H2O on perplexity/needle/decode-speed and reports the reward-model R² lift).
- Validation done here (numpy, no MLX): rollout past/future windowing matches an independent brute-force reference; constant-attention sanity gives avg_attention == label == c. **Off-by-one fixed**: `past` spans `age+1` contributing queries (token attends to itself at entry), so training now divides by `age+1` to match the inference normalization (`offset − position`) exactly — otherwise the StandardScaler mean/std would have biased inference.
- Honest modeling limitation (noted for interview): rollouts sample each token at a single fixed age, so the policy is trained on one lifetime slice; features are lifetime-normalized so the meaning generalizes, but this is a simplification the verdict accounts for.

## What Phase 4 built
- `longcache/master_benchmark.py` + `scripts/stress.py` — one master comparative table: all 7 methods (fp16 / int8 / int4 / recency / streaming / heavy_hitter / learned) at each context length, sharing a single loaded model and a single trained policy. Metrics: peak mem, KV mem, decode tok/s, TTFT, perplexity, perplexity Δ vs the fp16 baseline at that context, needle accuracy.
- Each (context, method) measurement is isolated in try/except: an OOM/failure is recorded as `OOM/fail` and the sweep continues — so the baseline hitting its ceiling while compressed methods stay bounded shows up as real data, not a crash. Results are written to disk after every context, so a multi-hour run is never lost.
- Reuses the validated low-level helpers (GenerationRun, cached_perplexity, run_needle_suite, HeavyHitterRunner, LearnedRunner) rather than re-implementing measurement.
- Validated here in pure Python: the perplexity-delta-vs-fp16 computation and the table's None/OOM cell handling (OOM rows show `OOM/fail` for memory, `—` for missing metrics, no spurious delta).
- Known cost: token-by-token methods (heavy_hitter, learned) dominate runtime at 16k; `stress.py` is the big run. Lower `context_lengths` / `needle_depths` in config.py for a faster pass.

## What Phase 5 built
- `longcache/charts.py` + `scripts/charts.py` — renders three figures from `results-raw/phase4_master.json` into `charts/`: memory-vs-context (baseline climbing to the measured OOM ceiling vs compressed methods staying bounded), needle-accuracy-vs-context, and a memory-vs-quality trade-off scatter at the reference context. Data extraction is matplotlib-free and was unit-tested here on synthetic results (incl. the OOM case); plotting imports matplotlib lazily. Chart script needs no MLX — only matplotlib + the results file — and refuses to run without real Phase 4 data (no placeholder charts).
- `scripts/linkedin.py` + `charts.py::headline_stats` — LinkedIn pack: renders the 3 charts and writes `LINKEDIN-<slug>.md`, a ready-to-paste post whose headline numbers (best compression ratio, perplexity delta, OOM ceiling, learned-vs-H2O verdict) are computed from the data, never hand-typed. `headline_stats` is OOM-aware: it measures the compression ratio at the largest context where the FP16 baseline actually ran, and reports the OOM ceiling + the longest context the compressed methods reached separately — so the story holds even on a low-RAM Mac where FP16 OOMs at 16k. Validated here on a synthetic OOM scenario (ref ctx 8k, ceiling 16k, honest "learned does not beat H2O" verdict). No screenshots needed — the charts are publication-ready PNGs.
- `master_benchmark.py` now records `budget_gb` / `unified_memory_gb` / `weight_gb` in the results so the chart can draw the OOM ceiling.
- `DEMO.md` — guided walkthrough (problem, four levers, run commands, what to look at, the honest engineering notes an interviewer will probe).
- README finalized: DEMO link, charts section + setup command, master-table anchor. RESUME_BULLETS finalized through Phase 4/5.
- `.gitignore` — `charts/` ignored until populated with real measured data; the real PNGs are committed (or attached to the post) after the Apple Silicon run.
- Honest status: Phase 5's "charts render from real benchmark data" criterion is met by the code, but the actual PNGs cannot exist until the Phase 4 run produces `phase4_master.json` on Apple Silicon. No fabricated charts are committed.

## Verified on this host
- All modules compile (`python3 -m py_compile`).
- `scripts/baseline.py` exits 2 on this Intel Mac with the correct hardware/Python diagnosis.
- mlx-lm API names checked against current upstream source (load, generate_step, make_prompt_cache, KVCache/QuantizedKVCache/RotatingKVCache, mx memory fns).

## Correctness audit (Phases 0–2, before first hardware run)
Traced every external mlx / mlx-lm call against upstream source to de-risk the first run on
borrowed hardware:
- Verified: `generate_step` accepts kv_bits/kv_group_size/quantized_kv_start (keyword-only) and
  yields `(token, logprobs)`; sampler is called on log-probs (argmax still correct); memory API
  `mx.get_peak_memory` / `mx.reset_peak_memory` / `mx.get_active_memory` are top-level;
  `nn.losses.cross_entropy(logits, targets, reduction=...)` signature; tokenizer `eos_token_ids`
  (set) with `eos_token_id` delegating to the HF tokenizer; `take/repeat/argsort/sort/`
  `concatenate/where` + `.astype/.transpose/.sum` all exist.
- BUG FOUND AND FIXED: `mx.softmax` does NOT accept `precise=` (would have crashed H2O capture
  with TypeError). Now upcasts to float32 explicitly. Also dropped reliance on `mx.arange(dtype=)`
  and `.swapaxes` (used `.transpose` + explicit `.astype`).
- Added `scripts/smoke.py`: ~30s end-to-end test that runs every path (baseline, INT8/INT4,
  recency/streaming/H2O, both perplexities, needle, and the H2O cache eviction/offset/score
  mechanics with real MLX ops) and asserts finite, ordered results. Run it FIRST on new
  hardware so any remaining integration issue fails in seconds, not mid-benchmark.

## Known issues / risks
- **Cannot measure anything here** — needs Apple Silicon. This is the single blocker to filling every table.
- `model.args` field names are introspected defensively; verify against the chosen model on first real run.
- Needle prompt length ≈ target context + question + chat-template overhead; harness records the actual measured prompt length.
- Held-out text is public-domain (may be in pretraining); perplexity is a relative baseline for measuring deltas, not unseen-data generalization.

## Next action
On an Apple Silicon Mac: `pip install -r requirements.txt`, `python scripts/get_data.py`,
`python scripts/smoke.py` (now also covers Phase 3: rollout + train + learned cache/runner),
then the phase scripts: `baseline.py` (0), `quantize.py` (1), `evict.py` (2), `learn.py` (3),
`stress.py` (4, the master table — the big run), `charts.py` (5, renders the figures). Note:
token-by-token methods (H2O + learned) are slow at 16k — expect minutes per context; lower
`context_lengths` / `eviction_context` / `rollout_context` / `needle_depths` in config.py for a
quick pass. Paste the printed tables back (and the two PNGs); I fill README / RESUME_BULLETS /
this file with the real numbers (baseline + OOM ceiling + FP16/INT8/INT4 deltas + eviction
comparison + learned-vs-heuristic verdict + master table) and commit the real charts. That
closes out all six phases — the only thing standing between this repo and a finished,
fully-measured portfolio project is one run on Apple Silicon.

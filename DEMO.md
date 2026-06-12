# LongCache — Demo & Walkthrough

A guided tour of what this project does, how to run it, and what to look at. For the
numbers-first summary see [README.md](README.md); for interview-ready bullets see
[RESUME_BULLETS.md](RESUME_BULLETS.md).

## The one-liner

On-device LLMs choke on long conversations because the KV cache grows linearly with every
token and must stay in RAM — a long chat slows down and eventually crashes with an
out-of-memory error. LongCache holds the longest possible conversation on a fixed memory
budget, on Apple Silicon with MLX, by compressing that cache (quantization + eviction) and
measuring exactly what each technique costs in quality.

## The hard part

The trade-off. Evict old tokens and the model forgets (quality drops); keep everything and it
crashes. Every claim here is backed by a measured number, never an estimate — and where a
technique makes things worse, the regression is reported honestly.

## The four levers (and how they're tested)

1. **Quantize what you keep** — store cached K/V in INT8/INT4 instead of FP16 (mlx-lm
   `QuantizedKVCache`). Quality is scored *through the quantized cache*, so the perplexity delta
   is the real quantized-attention cost, not an uncached approximation.
2. **Keep fewer tokens, heuristically** — recency window and StreamingLLM (attention sinks +
   recent) via mlx-lm `RotatingKVCache`; H2O heavy-hitter via a custom cache that recovers the
   per-token attention mass the fused attention kernel never exposes.
3. **Keep fewer tokens, learned** — frame keep/evict as a contextual bandit (features: attention
   received, position, value norm, layer; reward: future attention at fixed memory), train a
   linear reward model offline, deploy it as a cheap per-step scoring cache, and benchmark it
   head-to-head against the heuristics with an explicit verdict.
4. **Measure honestly** — peak/KV memory, throughput, TTFT, plus two offline quality metrics:
   perplexity on held-out long text and needle-in-a-haystack retrieval accuracy vs context.

## Run it (Apple Silicon, macOS 14+, Python 3.10+)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_data.py    # held-out long text (gitignored)
python scripts/smoke.py       # ~30s: every code path runs, fails fast if anything is off
python scripts/baseline.py    # Phase 0 — baseline + telemetry + OOM ceiling
python scripts/quantize.py    # Phase 1 — INT8/INT4 vs FP16
python scripts/evict.py       # Phase 2 — recency / StreamingLLM / H2O
python scripts/learn.py       # Phase 3 — learned policy vs heuristics + verdict
python scripts/stress.py      # Phase 4 — master comparative table (the big run)
python scripts/charts.py      # Phase 5 — render the two charts from the Phase 4 data
```

`smoke.py` first: it loads the model once and exercises baseline, quantization, all three
eviction strategies, both perplexity paths, the needle harness, and the learned-policy pipeline
with tiny inputs — so an integration issue surfaces in seconds, not 20 minutes into a benchmark.

## What to look at

- **The master table** (`scripts/stress.py`, in README) — every method × context length on peak
  memory, KV memory, tokens/sec, TTFT, perplexity Δ, and needle accuracy. One glance shows the
  memory/quality/speed trade-off.
- **memory_vs_context.png** — the baseline climbing to its OOM ceiling while the compressed
  engine stays bounded. This is the whole pitch in one picture.
- **needle_vs_context.png** — how much retrieval accuracy each compression method keeps as
  context grows.
- **memory_quality_tradeoff.png** — the memory/quality frontier; the best methods sit bottom-right.
- **The verdict** (`scripts/learn.py`) — does the learned policy actually beat H2O? Reported from
  the data either way; a learned policy that loses is a documented, valid result.

For sharing: `scripts/linkedin.py` renders the three charts and writes a `LINKEDIN-<machine>.md`
post draft with the headline numbers computed from the run (no screenshots needed — the charts are
publication-ready PNGs).

## The honest engineering notes (the part an interviewer will probe)

- **H2O and the learned policy prefill token-by-token.** The fused attention kernel hides the
  weights heavy-hitter needs, so capturing them means an un-fused, one-token-at-a-time pass. Their
  TTFT is therefore high — and the tables report it rather than hide it. Recency/StreamingLLM use
  the fast fused path.
- **The learned policy is deliberately linear** (standardize → dot product) so its decode-time
  cost stays small and measurable — that cost is exactly what the verdict weighs against any
  quality gain.
- **Perplexity is always offline**, against reference text — never "bounded" during free
  generation, which is not a meaningful quantity.

## Status

Phases 0–5 are code-complete, built against the verified current mlx-lm API, and validated as
far as a non-MLX host allows (compile, pure-Python logic checks, brute-force reference for the
rollout windowing). The measured numbers and rendered charts come from a run on Apple Silicon;
until then the tables read `TBD`, by design — the harness aborts on unsupported hardware rather
than emit a fabricated value.

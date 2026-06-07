# Resume bullets — LongCache

Interview-defensible bullets. Measured numbers only; nothing estimated. A bullet stays
`TBD` until the number behind it has been measured on Apple Silicon.

## Status
All six phases (baseline + telemetry, KV quantization, heuristic eviction, learned eviction,
master stress benchmark, docs + charts) code complete and API-verified; awaiting a run on an
Apple Silicon Mac to populate numbers and render charts. The build host is an Intel Mac, on
which MLX cannot run, so no metric is measured yet.

## Draft bullets (numbers pending Apple Silicon run)
- Built an on-device long-context LLM inference engine on Apple MLX that holds long
  conversations on a fixed unified-memory budget by compressing the KV cache (quantization +
  eviction), targeting Apple Silicon.
- Engineered a measurement harness reporting genuinely measured Time-to-First-Token,
  sustained decode tokens/sec, peak unified memory, and live KV-cache bytes as context grows,
  plus two offline quality metrics: sliding-window perplexity and needle-in-a-haystack
  retrieval accuracy vs context length.
- Established the baseline memory ceiling: identified the context length at which the KV cache
  pushes peak memory past the device budget (OOM ceiling) on a TBD-GB machine — TBD tokens.
- Quantized the KV cache to INT8/INT4 (mlx-lm `QuantizedKVCache`, group size 64), cutting
  measured KV memory by TBDx at 16k context for a TBD perplexity delta and TBD needle-accuracy
  change vs the FP16 baseline; measured quality through quantized attention, not an uncached
  proxy, so the delta is honest.
- Implemented three KV-eviction strategies on a fixed token budget — recency window and
  StreamingLLM via mlx-lm's `RotatingKVCache`, and H2O heavy-hitter via a custom cache that
  recovers per-token attention mass the fused attention kernel hides — and benchmarked all
  three against the full cache at 16k context on memory, perplexity delta, needle accuracy,
  and decode throughput (deltas TBD pending the Apple Silicon run).
- Framed KV eviction as a contextual bandit (features: attention received, position, value
  norm, layer; reward: future attention at fixed memory), collected rollouts from the model,
  trained a linear reward model offline, and deployed it as a cheap per-step dot-product cache
  policy benchmarked head-to-head against the H2O/StreamingLLM heuristics — with an explicit,
  data-driven verdict on whether learning the importance function beats the heuristic (verdict
  TBD pending the Apple Silicon run; an honest negative result is reported if it loses).
- Produced a single master comparative table (baseline vs INT8/INT4 vs recency/StreamingLLM/H2O
  vs learned) across context lengths on one shared model, with per-(context, method) isolation
  so a method that exhausts memory is recorded and the sweep continues — quantifying the
  memory/quality/throughput trade-off and the baseline's OOM ceiling on real hardware (TBD).
- Visualized the result in two charts generated from the measured data: peak memory vs context
  (baseline climbing to its OOM ceiling while the compressed engine stays bounded) and needle
  retrieval accuracy vs context — the memory/quality trade-off in two pictures.

## Honesty note for the interview
Numbers above are placeholders until measured. The harness aborts on non-Apple-Silicon
hardware instead of emitting a fabricated value; perplexity is computed offline against
reference text, never "bounded" during live generation.

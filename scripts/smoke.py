#!/usr/bin/env python3
"""Fast end-to-end smoke test: exercise every code path with tiny inputs in ~30s.

Run this FIRST on Apple Silicon, before the full benchmarks. It loads the model once and
drives baseline generation, FP16/INT8/INT4 quantized caches, recency/StreamingLLM/H2O
eviction, both perplexity paths, the needle harness, and the H2O cache mechanics directly —
asserting no exceptions and finite, ordered results. A long benchmark failing 20 minutes in
is expensive; this fails (or passes) in seconds.
"""

import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longcache.preflight import preflight

preflight()

import mlx.core as mx

from longcache.cached_perplexity import cached_perplexity
from longcache.config import BaselineConfig
from longcache.generation import GenerationRun
from longcache.heavy_hitter import HeavyHitterCache, HeavyHitterRunner
from longcache.model_runtime import ModelRuntime
from longcache.needle import NeedleHaystack, run_needle_suite
from longcache.perplexity import sliding_window_perplexity
from longcache.textdata import HoldoutText


class Checks:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def run(self, name, fn):
        try:
            detail = fn()
            self.passed += 1
            print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))
        except Exception as exc:
            self.failed += 1
            print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")
            traceback.print_exc()


def _finite_ppl(result):
    ppl = result["perplexity"]
    assert math.isfinite(ppl) and ppl > 1.0, f"perplexity not sane: {ppl}"
    return ppl


def main():
    ctx = 96
    budget, sink, recent = 48, 4, 24
    decode = 6
    chunk = 32

    config = BaselineConfig()
    runtime = ModelRuntime(config)
    checks = Checks()

    def load_model():
        runtime.load()
        _assert(runtime.model is not None, "load returned no model")
        _assert(runtime.dims.num_layers > 0, "no layers")
        return config.model_id

    checks.run("load model", load_model)
    if runtime.model is None:
        print("\nmodel failed to load; aborting smoke test.")
        raise SystemExit(1)

    def fit_check():
        runtime.assert_fits()
        return f"weights {runtime.weight_gb():.2f} GB / budget {runtime.budget_gb():.2f} GB"

    checks.run("fit check", fit_check)

    holdout = HoldoutText(config.holdout_path)
    ids = holdout.token_ids(runtime.tokenizer)

    def holdout_check():
        _assert(len(ids) >= ctx * 2, f"only {len(ids)} tokens; need >= {ctx * 2}")
        return f"{len(ids)} tokens"

    checks.run("holdout has enough tokens", holdout_check)

    runner = GenerationRun(runtime)
    prompt = holdout.prompt_array(runtime.tokenizer, ctx)

    full_kv = {}

    def baseline_gen():
        out = runner.run(prompt, decode, stop_on_eos=False)
        _assert(out["decoded_tokens"] > 0, "no tokens decoded")
        _assert(out["kv_memory_gb"] > 0, "kv bytes zero")
        _assert(out["peak_memory_gb"] > 0, "peak memory zero")
        _assert(out["ttft_s"] is not None and out["ttft_s"] >= 0, "bad ttft")
        full_kv["fp16"] = out["kv_memory_gb"]
        return f"{out['decoded_tokens']} tok, kv {out['kv_memory_gb']*1024:.1f} MB, {out['decode_tokens_per_s']:.1f} tok/s"

    checks.run("baseline generation (full cache)", baseline_gen)
    checks.run("cached perplexity FP16", lambda: f"ppl {_finite_ppl(cached_perplexity(runtime.model, ids[:ctx], runtime.new_cache(), chunk)):.2f}")
    checks.run("sliding-window perplexity", lambda: f"ppl {_finite_ppl(sliding_window_perplexity(runtime.model, ids[:ctx], 64, 32)):.2f}")

    def quant_gen(bits):
        out = runner.run(prompt, decode, stop_on_eos=False, kv_bits=bits)
        _assert(out["kv_memory_gb"] > 0, "kv bytes zero")
        _assert(out["kv_memory_gb"] < full_kv["fp16"] * 1.05, f"int{bits} kv not smaller than fp16")
        full_kv[bits] = out["kv_memory_gb"]
        return f"kv {out['kv_memory_gb']*1024:.1f} MB ({full_kv['fp16']/out['kv_memory_gb']:.2f}x vs fp16)"

    checks.run("INT8 quantized generation", lambda: quant_gen(8))
    checks.run("INT4 quantized generation", lambda: quant_gen(4))
    checks.run("INT4 < INT8 < FP16 KV bytes", lambda: _assert(full_kv.get(4, 9) < full_kv.get(8, 9) < full_kv["fp16"], f"ordering off: {full_kv}") or "ok")
    checks.run("cached perplexity INT8", lambda: f"ppl {_finite_ppl(cached_perplexity(runtime.model, ids[:ctx], runtime.quantized_cache(8), chunk)):.2f}")
    checks.run("cached perplexity INT4", lambda: f"ppl {_finite_ppl(cached_perplexity(runtime.model, ids[:ctx], runtime.quantized_cache(4), chunk)):.2f}")

    def evict_path(keep):
        gen = runner.run(prompt, decode, prompt_cache=runtime.rotating_cache(budget, keep), stop_on_eos=False)
        _assert(gen["kv_memory_gb"] < full_kv["fp16"], "rotating cache not bounded below full")
        ppl = cached_perplexity(runtime.model, ids[:ctx], runtime.rotating_cache(budget, keep), chunk)
        return f"kv {gen['kv_memory_gb']*1024:.1f} MB, ppl {_finite_ppl(ppl):.2f}"

    checks.run("recency window (RotatingKVCache keep=0)", lambda: evict_path(0))
    checks.run("StreamingLLM (RotatingKVCache keep=sink)", lambda: evict_path(sink))

    def h2o_cache_unit():
        cache = HeavyHitterCache(budget=8, sink=2, recent=3)
        shape = (1, runtime.dims.num_kv_heads, 1, runtime.dims.head_dim)
        for _ in range(20):
            cache.update_and_fetch(mx.random.normal(shape), mx.random.normal(shape))
            cache.record_scores(mx.ones((cache.keys.shape[2],), dtype=mx.float32))
        _assert(cache.keys.shape[2] == 8, f"not capped at budget: {cache.keys.shape[2]}")
        _assert(cache.offset == 20, f"offset wrong: {cache.offset}")
        _assert(cache.scores.shape[0] == 8, "scores misaligned with keys")
        return "retained capped at budget=8, offset=20, scores aligned"

    checks.run("H2O cache mechanics (eviction/offset/scores)", h2o_cache_unit)

    h2o = HeavyHitterRunner(runtime, budget, sink, recent)
    checks.run("H2O generation (token-wise + attention capture)", lambda: (_assert(len(h2o.generate(prompt[:48], decode)["token_ids"]) > 0, "no tokens"), "ran")[1])
    checks.run("H2O perplexity (token-wise)", lambda: f"ppl {_finite_ppl(h2o.perplexity(ids[:48])):.2f}")

    filler = holdout.filler_text(runtime.tokenizer, ctx * 2)
    haystack = NeedleHaystack(filler, runtime.tokenizer, config.seed)
    checks.run("needle harness (full cache)", lambda: f"acc {run_needle_suite(runner, haystack, ctx, [0.5], 8)['accuracy']:.2f}")
    checks.run("needle harness (rotating cache_factory)", lambda: f"acc {run_needle_suite(runner, haystack, ctx, [0.5], 8, cache_factory=lambda: runtime.rotating_cache(budget, sink))['accuracy']:.2f}")

    print(f"\n{checks.passed} passed, {checks.failed} failed.")
    if checks.failed:
        print("Smoke test FAILED — fix before running the full benchmarks.")
        raise SystemExit(1)
    print("Smoke test PASSED — all code paths run on this machine. Safe to run the benchmarks.")


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)
    return True


if __name__ == "__main__":
    main()

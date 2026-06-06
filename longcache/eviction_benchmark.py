"""Phase 2: heuristic KV eviction compared against the full cache at a fixed context.

Three budget-limited strategies hold the same number of tokens and are scored against the
full (keep-everything) cache:
  - recency        RotatingKVCache(keep=0): a pure sliding window.
  - streaming      RotatingKVCache(keep=sink): StreamingLLM, attention sinks + recent window.
  - heavy_hitter   H2O: keep attention sinks + recent window + the highest-attention tokens.
"""

import json
from pathlib import Path

from .cached_perplexity import cached_perplexity
from .generation import GenerationRun
from .heavy_hitter import HeavyHitterRunner
from .model_runtime import ModelRuntime
from .needle import NeedleHaystack, run_needle_suite
from .textdata import HoldoutText


class EvictionBenchmark:
    def __init__(self, config):
        self.config = config
        self.runtime = ModelRuntime(config)
        self.holdout = None
        self.runner = None
        self.h2o = None

    def setup(self):
        self.runtime.load()
        report = self.runtime.assert_fits()
        self.holdout = HoldoutText(self.config.holdout_path)
        self.runner = GenerationRun(self.runtime)
        self.h2o = HeavyHitterRunner(
            self.runtime,
            self.config.eviction_budget,
            self.config.heavy_sink,
            self.config.heavy_recent,
        )
        return report

    def _haystack(self, context_length):
        filler = self.holdout.filler_text(self.runtime.tokenizer, context_length * 2)
        return NeedleHaystack(filler, self.runtime.tokenizer, self.config.seed)

    def _measure_cache(self, name, context_length, cache_factory):
        tokenizer = self.runtime.tokenizer
        prompt_ids = self.holdout.prompt_array(tokenizer, context_length)
        generation = self.runner.run(
            prompt_ids,
            self.config.decode_tokens,
            prompt_cache=cache_factory(),
            stop_on_eos=False,
        )

        ppl_ids = self.holdout.token_ids(tokenizer)[:context_length]
        perplexity = cached_perplexity(
            self.runtime.model, ppl_ids, cache_factory(), self.config.quant_chunk_size
        )

        needle = run_needle_suite(
            self.runner,
            self._haystack(context_length),
            context_length,
            self.config.needle_depths,
            self.config.needle_answer_tokens,
            cache_factory=cache_factory,
        )
        return self._row(name, generation, perplexity, needle)

    def _measure_heavy_hitter(self, context_length):
        tokenizer = self.runtime.tokenizer
        prompt_ids = self.holdout.prompt_array(tokenizer, context_length)
        generation = self.h2o.generate(
            prompt_ids, self.config.decode_tokens, stop_on_eos=False
        )

        ppl_ids = self.holdout.token_ids(tokenizer)[:context_length]
        perplexity = self.h2o.perplexity(ppl_ids)

        haystack = self._haystack(context_length)
        correct = 0
        records = []
        for depth in self.config.needle_depths:
            prompt, secret = haystack.build(context_length, depth)
            out = self.h2o.generate(prompt, self.config.needle_answer_tokens)
            ok = NeedleHaystack.is_correct(out["text"], secret)
            correct += int(ok)
            records.append({"depth": depth, "secret": secret, "correct": ok})
        needle = {"accuracy": correct / len(self.config.needle_depths), "records": records}
        return self._row("heavy_hitter", generation, perplexity, needle)

    def _row(self, name, generation, perplexity, needle):
        return {
            "strategy": name,
            "kv_memory_gb": generation["kv_memory_gb"],
            "peak_memory_gb": generation["peak_memory_gb"],
            "perplexity": perplexity["perplexity"],
            "needle_accuracy": needle["accuracy"],
            "decode_tokens_per_s": generation["decode_tokens_per_s"],
            "ttft_s": generation["ttft_s"],
        }

    def run(self):
        if self.runner is None:
            self.setup()
        ctx = self.config.eviction_context
        budget = self.config.eviction_budget
        rows = [
            self._measure_cache("full", ctx, lambda: self.runtime.new_cache()),
            self._measure_cache(
                "recency", ctx, lambda: self.runtime.rotating_cache(budget, 0)
            ),
            self._measure_cache(
                "streaming",
                ctx,
                lambda: self.runtime.rotating_cache(budget, self.config.streaming_sink),
            ),
            self._measure_heavy_hitter(ctx),
        ]
        return {
            "model_id": self.config.model_id,
            "context_length": ctx,
            "budget": budget,
            "rows": rows,
        }

    def save(self, results):
        out_dir = Path(self.config.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "phase2_eviction.json"
        path.write_text(json.dumps(results, indent=2, default=str))
        return path


def _fmt(value, spec):
    if value is None:
        return "TBD"
    return format(value, spec)


def eviction_table(results):
    rows = results["rows"]
    full = next((r for r in rows if r["strategy"] == "full"), {})
    full_kv = full.get("kv_memory_gb")
    full_ppl = full.get("perplexity")

    header = (
        "| Strategy | KV mem (GB) | KV vs full | Peak mem (GB) | Perplexity | "
        "PPL Δ vs full | Needle acc. | Decode tok/s | TTFT (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        kv_ratio = None
        if full_kv and row["kv_memory_gb"]:
            kv_ratio = full_kv / row["kv_memory_gb"]
        ppl_delta = None
        if full_ppl is not None and row["perplexity"] is not None:
            ppl_delta = row["perplexity"] - full_ppl
        is_full = row["strategy"] == "full"
        lines.append(
            "| {name} | {kv} | {ratio} | {peak} | {ppl} | {dppl} | {needle} | {tps} | {ttft} |".format(
                name=row["strategy"],
                kv=_fmt(row["kv_memory_gb"], ".3f"),
                ratio=("1.00x (ref)" if is_full else (f"{kv_ratio:.2f}x" if kv_ratio else "TBD")),
                peak=_fmt(row["peak_memory_gb"], ".2f"),
                ppl=_fmt(row["perplexity"], ".2f"),
                dppl=("0.00 (ref)" if is_full else _fmt(ppl_delta, "+.2f")),
                needle=_fmt(row["needle_accuracy"], ".2f"),
                tps=_fmt(row["decode_tokens_per_s"], ".1f"),
                ttft=_fmt(row["ttft_s"], ".2f"),
            )
        )
    return "\n".join(lines)

"""Phase 4: stress benchmark producing one master comparative table across all methods.

Runs every compression method at each context length, sharing a single loaded model, and
assembles one table: peak memory, KV memory, tokens/sec, TTFT, perplexity (+ delta vs the FP16
baseline), and needle accuracy. Each (context, method) measurement is isolated so a method that
runs out of memory at a given context is recorded as a failure and the sweep continues — that
is the whole point: the full baseline climbs to its OOM ceiling while the compressed methods
stay bounded. Results are written to disk after every context so a long run is never lost.
"""

import json
from pathlib import Path

from .cached_perplexity import cached_perplexity
from .generation import GenerationRun
from .heavy_hitter import HeavyHitterRunner
from .learned_eviction import LearnedRunner, collect_rollouts, train_policy
from .model_runtime import ModelRuntime
from .needle import NeedleHaystack, run_needle_suite
from .textdata import HoldoutText

METHODS = ["fp16", "int8", "int4", "recency", "streaming", "heavy_hitter", "learned"]


class MasterBenchmark:
    def __init__(self, config):
        self.config = config
        self.runtime = ModelRuntime(config)
        self.holdout = None
        self.runner = None
        self.h2o = None
        self.learned_runner = None
        self.policy = None

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

    def train_learned_policy(self):
        token_ids = self.holdout.token_ids(self.runtime.tokenizer)[
            : self.config.rollout_context
        ]
        features, targets = collect_rollouts(
            self.runtime, token_ids, self.config.bandit_age, self.config.bandit_future
        )
        self.policy = train_policy(
            features, targets, self.runtime, self.config.bandit_age, self.config.bandit_future
        )
        self.policy.to_json(self.config.policy_path)
        self.learned_runner = LearnedRunner(
            self.runtime,
            self.policy,
            self.config.eviction_budget,
            self.config.heavy_sink,
            self.config.heavy_recent,
        )
        return self.policy

    def _haystack(self, context_length):
        filler = self.holdout.filler_text(self.runtime.tokenizer, context_length * 2)
        return NeedleHaystack(filler, self.runtime.tokenizer, self.config.seed)

    def _row(self, context_length, method, status, generation, perplexity, needle):
        return {
            "context_length": context_length,
            "method": method,
            "status": status,
            "peak_memory_gb": generation.get("peak_memory_gb") if generation else None,
            "kv_memory_gb": generation.get("kv_memory_gb") if generation else None,
            "decode_tokens_per_s": generation.get("decode_tokens_per_s") if generation else None,
            "ttft_s": generation.get("ttft_s") if generation else None,
            "perplexity": perplexity.get("perplexity") if perplexity else None,
            "perplexity_delta": None,
            "needle_accuracy": needle.get("accuracy") if needle else None,
        }

    def _cache_method(self, context_length, method, gen, ppl_factory, needle_kwargs):
        tokenizer = self.runtime.tokenizer
        prompt_ids = self.holdout.prompt_array(tokenizer, context_length)
        generation = gen(prompt_ids)
        ppl_ids = self.holdout.token_ids(tokenizer)[:context_length]
        perplexity = cached_perplexity(
            self.runtime.model, ppl_ids, ppl_factory(), self.config.quant_chunk_size
        )
        needle = run_needle_suite(
            self.runner,
            self._haystack(context_length),
            context_length,
            self.config.needle_depths,
            self.config.needle_answer_tokens,
            **needle_kwargs,
        )
        return self._row(context_length, method, "ok", generation, perplexity, needle)

    def _stream_method(self, context_length, method, runner):
        tokenizer = self.runtime.tokenizer
        prompt_ids = self.holdout.prompt_array(tokenizer, context_length)
        generation = runner.generate(
            prompt_ids, self.config.decode_tokens, stop_on_eos=False
        )
        ppl_ids = self.holdout.token_ids(tokenizer)[:context_length]
        perplexity = runner.perplexity(ppl_ids)

        haystack = self._haystack(context_length)
        correct = 0
        for depth in self.config.needle_depths:
            prompt, secret = haystack.build(context_length, depth)
            out = runner.generate(prompt, self.config.needle_answer_tokens)
            correct += int(NeedleHaystack.is_correct(out["text"], secret))
        needle = {"accuracy": correct / len(self.config.needle_depths)}
        return self._row(context_length, method, "ok", generation, perplexity, needle)

    def _measure(self, context_length, method):
        decode = self.config.decode_tokens
        group = self.config.kv_group_size
        budget = self.config.eviction_budget
        try:
            if method in ("fp16", "int8", "int4"):
                kv_bits = None if method == "fp16" else int(method[3:])
                return self._cache_method(
                    context_length,
                    method,
                    gen=lambda p: self.runner.run(
                        p, decode, stop_on_eos=False, kv_bits=kv_bits, kv_group_size=group
                    ),
                    ppl_factory=(
                        (lambda: self.runtime.quantized_cache(kv_bits, group))
                        if kv_bits is not None
                        else self.runtime.new_cache
                    ),
                    needle_kwargs={"kv_bits": kv_bits},
                )
            if method in ("recency", "streaming"):
                keep = 0 if method == "recency" else self.config.streaming_sink
                factory = lambda: self.runtime.rotating_cache(budget, keep)
                return self._cache_method(
                    context_length,
                    method,
                    gen=lambda p: self.runner.run(
                        p, decode, prompt_cache=factory(), stop_on_eos=False
                    ),
                    ppl_factory=factory,
                    needle_kwargs={"cache_factory": factory},
                )
            if method == "heavy_hitter":
                return self._stream_method(context_length, method, self.h2o)
            if method == "learned":
                return self._stream_method(context_length, method, self.learned_runner)
            raise ValueError(f"unknown method {method}")
        except (MemoryError, RuntimeError, ValueError) as exc:
            return self._row(
                context_length, method, f"failed: {type(exc).__name__}: {exc}", None, None, None
            )

    def run(self):
        self.setup()
        self.train_learned_policy()
        rows = []
        for context_length in self.config.context_lengths:
            context_rows = [self._measure(context_length, m) for m in METHODS]
            baseline = next((r for r in context_rows if r["method"] == "fp16"), None)
            base_ppl = baseline["perplexity"] if baseline else None
            if base_ppl is not None:
                for row in context_rows:
                    if row["perplexity"] is not None:
                        row["perplexity_delta"] = row["perplexity"] - base_ppl
            rows.extend(context_rows)
            self.save(self._results(rows))
        return self._results(rows)

    def _results(self, rows):
        return {
            "model_id": self.config.model_id,
            "context_lengths": self.config.context_lengths,
            "budget": self.config.eviction_budget,
            "policy": self.policy.summary() if self.policy else None,
            "rows": rows,
        }

    def save(self, results):
        out_dir = Path(self.config.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "phase4_master.json"
        path.write_text(json.dumps(results, indent=2, default=str))
        return path


def _fmt(value, spec):
    if value is None:
        return "—"
    return format(value, spec)


def master_table(results):
    header = (
        "| Context | Method | Peak mem (GB) | KV mem (GB) | Tokens/sec | TTFT (s) | "
        "Perplexity | PPL Δ | Needle acc. |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in results["rows"]:
        failed = row["status"] != "ok"
        note = "OOM/fail" if failed else None
        lines.append(
            "| {ctx} | {method} | {peak} | {kv} | {tps} | {ttft} | {ppl} | {dppl} | {needle} |".format(
                ctx=row["context_length"],
                method=row["method"],
                peak=(note or _fmt(row["peak_memory_gb"], ".2f")),
                kv=(note or _fmt(row["kv_memory_gb"], ".3f")),
                tps=_fmt(row["decode_tokens_per_s"], ".1f"),
                ttft=_fmt(row["ttft_s"], ".2f"),
                ppl=_fmt(row["perplexity"], ".2f"),
                dppl=(
                    "0.00 (ref)"
                    if row["method"] == "fp16" and row["perplexity"] is not None
                    else _fmt(row["perplexity_delta"], "+.2f")
                ),
                needle=_fmt(row["needle_accuracy"], ".2f"),
            )
        )
    return "\n".join(lines)

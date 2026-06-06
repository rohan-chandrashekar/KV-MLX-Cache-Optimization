"""Phase 0 baseline orchestration: sweep context lengths, measure, emit tables."""

import json
from pathlib import Path

from .generation import GenerationRun
from .model_runtime import ModelRuntime
from .needle import NeedleHaystack, run_needle_suite
from .perplexity import sliding_window_perplexity
from .textdata import HoldoutText


class BaselineBenchmark:
    def __init__(self, config):
        self.config = config
        self.runtime = ModelRuntime(config)
        self.holdout = None
        self.runner = None

    def setup(self):
        self.runtime.load()
        report = self.runtime.assert_fits()
        self.holdout = HoldoutText(self.config.holdout_path)
        self.runner = GenerationRun(self.runtime)
        return report

    def measure_generation(self, context_length):
        prompt_ids = self.holdout.prompt_array(self.runtime.tokenizer, context_length)
        result = self.runner.run(
            prompt_ids, self.config.decode_tokens, stop_on_eos=False
        )
        return result

    def measure_perplexity(self, max_len):
        ids = self.holdout.token_ids(self.runtime.tokenizer)
        ids = ids[: self.config.perplexity_token_budget]
        return sliding_window_perplexity(
            self.runtime.model, ids, max_len, self.config.perplexity_stride
        )

    def measure_needle(self, context_length):
        filler = self.holdout.filler_text(
            self.runtime.tokenizer, context_length * 2
        )
        haystack = NeedleHaystack(filler, self.runtime.tokenizer, self.config.seed)
        return run_needle_suite(
            self.runner,
            haystack,
            context_length,
            self.config.needle_depths,
            self.config.needle_answer_tokens,
        )

    def find_oom_ceiling(self):
        budget = self.runtime.budget_gb()
        ceiling = None
        length = self.config.oom_probe_step
        while length <= self.config.oom_probe_max:
            try:
                result = self.measure_generation(length)
            except (MemoryError, RuntimeError, ValueError) as exc:
                return {"ceiling_tokens": ceiling, "stopped_at": length, "reason": str(exc)}
            if result["peak_memory_gb"] > budget:
                return {
                    "ceiling_tokens": ceiling,
                    "stopped_at": length,
                    "reason": f"peak {result['peak_memory_gb']:.2f} GB exceeded budget {budget:.2f} GB",
                }
            ceiling = length
            length += self.config.oom_probe_step
        return {"ceiling_tokens": ceiling, "stopped_at": None, "reason": "probe_max reached"}

    def run(self, with_oom=True):
        fit = self.setup()
        rows = []
        for length in self.config.context_lengths:
            generation = self.measure_generation(length)
            perplexity = self.measure_perplexity(min(length, self.config.perplexity_max_len))
            needle = self.measure_needle(length)
            rows.append(
                {
                    "context_length": length,
                    "ttft_s": generation["ttft_s"],
                    "decode_tokens_per_s": generation["decode_tokens_per_s"],
                    "peak_memory_gb": generation["peak_memory_gb"],
                    "kv_memory_gb": generation["kv_memory_gb"],
                    "perplexity": perplexity["perplexity"],
                    "needle_accuracy": needle["accuracy"],
                    "needle_detail": needle["records"],
                }
            )
        oom = self.find_oom_ceiling() if with_oom else None
        return {
            "model_id": self.config.model_id,
            "fit_report": _jsonable(fit),
            "rows": rows,
            "oom": oom,
        }

    def save(self, results):
        out_dir = Path(self.config.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "phase0_baseline.json"
        path.write_text(json.dumps(results, indent=2, default=str))
        return path


def _jsonable(report):
    out = dict(report)
    dims = out.get("dims")
    if dims is not None:
        out["dims"] = vars(dims)
    return out


def _fmt(value, spec):
    if value is None:
        return "TBD"
    return format(value, spec)


def baseline_table(results):
    header = (
        "| Context (tokens) | TTFT (s) | Decode tok/s | Peak memory (GB) | "
        "KV memory (GB) | Perplexity | Needle acc. |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in results["rows"]:
        lines.append(
            "| {ctx} | {ttft} | {tps} | {peak} | {kv} | {ppl} | {needle} |".format(
                ctx=row["context_length"],
                ttft=_fmt(row["ttft_s"], ".3f"),
                tps=_fmt(row["decode_tokens_per_s"], ".1f"),
                peak=_fmt(row["peak_memory_gb"], ".2f"),
                kv=_fmt(row["kv_memory_gb"], ".3f"),
                ppl=_fmt(row["perplexity"], ".2f"),
                needle=_fmt(row["needle_accuracy"], ".2f"),
            )
        )
    oom = results.get("oom") or {}
    ceiling = oom.get("ceiling_tokens")
    lines.append("")
    lines.append(
        f"OOM ceiling (largest context under memory budget): "
        f"{ceiling if ceiling is not None else 'TBD'} tokens"
        + (f" — {oom.get('reason')}" if oom.get("reason") else "")
    )
    return "\n".join(lines)

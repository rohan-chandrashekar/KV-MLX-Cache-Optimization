"""Phase 1: KV-cache quantization (INT8 / INT4) benchmarked against the FP16 baseline."""

import json
from pathlib import Path

from .cached_perplexity import cached_perplexity
from .generation import GenerationRun
from .model_runtime import ModelRuntime
from .needle import NeedleHaystack, run_needle_suite
from .textdata import HoldoutText


def precision_bits(precision):
    return None if precision == "fp16" else int(precision)


def precision_label(precision):
    return "fp16" if precision == "fp16" else f"int{precision}"


class QuantizationBenchmark:
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

    def measure(self, context_length, precision):
        kv_bits = precision_bits(precision)
        tokenizer = self.runtime.tokenizer

        prompt_ids = self.holdout.prompt_array(tokenizer, context_length)
        generation = self.runner.run(
            prompt_ids,
            self.config.decode_tokens,
            stop_on_eos=False,
            kv_bits=kv_bits,
            kv_group_size=self.config.kv_group_size,
        )

        ppl_ids = self.holdout.token_ids(tokenizer)[:context_length]
        cache = self.runtime.quantized_cache(kv_bits, self.config.kv_group_size)
        perplexity = cached_perplexity(
            self.runtime.model, ppl_ids, cache, self.config.quant_chunk_size
        )

        filler = self.holdout.filler_text(tokenizer, context_length * 2)
        haystack = NeedleHaystack(filler, tokenizer, self.config.seed)
        needle = run_needle_suite(
            self.runner,
            haystack,
            context_length,
            self.config.needle_depths,
            self.config.needle_answer_tokens,
            kv_bits=kv_bits,
        )

        return {
            "context_length": context_length,
            "precision": precision_label(precision),
            "kv_memory_gb": generation["kv_memory_gb"],
            "peak_memory_gb": generation["peak_memory_gb"],
            "perplexity": perplexity["perplexity"],
            "needle_accuracy": needle["accuracy"],
            "decode_tokens_per_s": generation["decode_tokens_per_s"],
            "ttft_s": generation["ttft_s"],
        }

    def run(self):
        self.setup()
        rows = []
        for context_length in self.config.context_lengths:
            for precision in self.config.kv_precisions:
                rows.append(self.measure(context_length, precision))
        return {"model_id": self.config.model_id, "rows": rows}

    def save(self, results):
        out_dir = Path(self.config.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "phase1_quantization.json"
        path.write_text(json.dumps(results, indent=2, default=str))
        return path


def _fmt(value, spec):
    if value is None:
        return "TBD"
    return format(value, spec)


def _baseline_lookup(rows):
    baseline = {}
    for row in rows:
        if row["precision"] == "fp16":
            baseline[row["context_length"]] = row
    return baseline


def quantization_table(results):
    rows = results["rows"]
    baseline = _baseline_lookup(rows)
    header = (
        "| Context | Precision | KV mem (GB) | KV vs FP16 | Peak mem (GB) | "
        "Perplexity | PPL Δ | Needle acc. | Decode tok/s |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        base = baseline.get(row["context_length"], {})
        base_kv = base.get("kv_memory_gb")
        base_ppl = base.get("perplexity")

        kv_ratio = None
        if base_kv and row["kv_memory_gb"]:
            kv_ratio = base_kv / row["kv_memory_gb"]
        ppl_delta = None
        if base_ppl is not None and row["perplexity"] is not None:
            ppl_delta = row["perplexity"] - base_ppl

        lines.append(
            "| {ctx} | {prec} | {kv} | {ratio} | {peak} | {ppl} | {dppl} | {needle} | {tps} |".format(
                ctx=row["context_length"],
                prec=row["precision"],
                kv=_fmt(row["kv_memory_gb"], ".3f"),
                ratio=(f"{kv_ratio:.2f}x" if kv_ratio is not None else "TBD"),
                peak=_fmt(row["peak_memory_gb"], ".2f"),
                ppl=_fmt(row["perplexity"], ".2f"),
                dppl=(
                    "0.00 (ref)"
                    if row["precision"] == "fp16"
                    else _fmt(ppl_delta, "+.2f")
                ),
                needle=_fmt(row["needle_accuracy"], ".2f"),
                tps=_fmt(row["decode_tokens_per_s"], ".1f"),
            )
        )
    return "\n".join(lines)

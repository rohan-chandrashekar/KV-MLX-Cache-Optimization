#!/usr/bin/env python3
"""Phase 1 entrypoint: KV-cache quantization sweep (FP16 vs INT8 vs INT4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longcache.preflight import preflight

preflight()

from longcache.config import BaselineConfig
from longcache.quantization_benchmark import QuantizationBenchmark, quantization_table


def main():
    config = BaselineConfig()
    benchmark = QuantizationBenchmark(config)

    fit = benchmark.setup()
    print("== Fit report ==")
    print(f"model: {fit['model_id']}")
    print(f"weights: {fit['weight_gb']:.2f} GB")
    print(f"budget ({config.memory_budget_fraction:.0%}): {fit['budget_gb']:.2f} GB")
    print()

    rows = []
    for context_length in config.context_lengths:
        for precision in config.kv_precisions:
            rows.append(benchmark.measure(context_length, precision))
    results = {"model_id": config.model_id, "rows": rows}
    path = benchmark.save(results)

    print("== Phase 1 KV-cache quantization ==")
    print(quantization_table(results))
    print()
    print(f"raw results written to {path}")


if __name__ == "__main__":
    main()

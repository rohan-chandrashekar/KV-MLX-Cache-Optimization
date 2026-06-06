#!/usr/bin/env python3
"""Phase 0 entrypoint: run the baseline sweep and print the measured table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longcache.preflight import preflight

preflight()

from longcache.benchmark import BaselineBenchmark, baseline_table
from longcache.config import BaselineConfig


def main():
    config = BaselineConfig()
    benchmark = BaselineBenchmark(config)

    fit = benchmark.setup()
    print("== Fit report ==")
    print(f"model: {fit['model_id']}")
    print(f"weights: {fit['weight_gb']:.2f} GB")
    print(f"unified memory: {fit['unified_memory_gb']:.2f} GB")
    print(f"budget ({config.memory_budget_fraction:.0%}): {fit['budget_gb']:.2f} GB")
    print(
        f"analytical KV @ {max(config.context_lengths)} tokens: "
        f"{fit['analytical_kv_gb_at_max_ctx']:.2f} GB"
    )
    print(f"projected peak @ max ctx: {fit['projected_gb_at_max_ctx']:.2f} GB")
    print()

    results = benchmark.run(with_oom=True)
    path = benchmark.save(results)

    print("== Phase 0 baseline ==")
    print(baseline_table(results))
    print()
    print(f"raw results written to {path}")


if __name__ == "__main__":
    main()

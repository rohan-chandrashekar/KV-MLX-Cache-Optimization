#!/usr/bin/env python3
"""Phase 4 entrypoint: stress benchmark + master comparative table across all methods."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longcache.preflight import preflight

preflight()

from longcache.config import BaselineConfig
from longcache.master_benchmark import METHODS, MasterBenchmark, master_table


def main():
    config = BaselineConfig()
    benchmark = MasterBenchmark(config)

    fit = benchmark.setup()
    print("== Fit report ==")
    print(f"model: {fit['model_id']}")
    print(f"context sweep: {config.context_lengths}")
    print(f"eviction budget: {config.eviction_budget} tokens")
    print()

    print("== Training learned policy ==")
    policy = benchmark.train_learned_policy()
    print(f"rows: {policy.rows} | train R²: {policy.train_r2:.4f} | baseline R²: {policy.baseline_r2:.4f}")
    print()

    rows = []
    for context_length in config.context_lengths:
        print(f"-- context {context_length} --")
        context_rows = [benchmark._measure(context_length, m) for m in METHODS]
        baseline = next((r for r in context_rows if r["method"] == "fp16"), None)
        base_ppl = baseline["perplexity"] if baseline else None
        if base_ppl is not None:
            for row in context_rows:
                if row["perplexity"] is not None:
                    row["perplexity_delta"] = row["perplexity"] - base_ppl
        for row in context_rows:
            print(f"   {row['method']:<13} {row['status']}")
        rows.extend(context_rows)
        benchmark.save(benchmark._results(rows))

    results = benchmark._results(rows)
    print()
    print("== Phase 4 master comparative table ==")
    print(master_table(results))
    print()
    print(f"raw results written to {Path(config.results_dir) / 'phase4_master.json'}")


if __name__ == "__main__":
    main()

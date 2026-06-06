#!/usr/bin/env python3
"""Phase 2 entrypoint: heuristic KV eviction (recency / StreamingLLM / H2O) vs full cache."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longcache.preflight import preflight

preflight()

from longcache.config import BaselineConfig
from longcache.eviction_benchmark import EvictionBenchmark, eviction_table


def main():
    config = BaselineConfig()
    benchmark = EvictionBenchmark(config)

    fit = benchmark.setup()
    print("== Fit report ==")
    print(f"model: {fit['model_id']}")
    print(f"context: {config.eviction_context} tokens, budget: {config.eviction_budget} tokens")
    print(
        f"streaming sink: {config.streaming_sink}, "
        f"H2O sink/recent: {config.heavy_sink}/{config.heavy_recent}"
    )
    print()

    results = benchmark.run()
    path = benchmark.save(results)

    print(f"== Phase 2 KV eviction @ {results['context_length']} tokens ==")
    print(eviction_table(results))
    print()
    print(f"raw results written to {path}")


if __name__ == "__main__":
    main()

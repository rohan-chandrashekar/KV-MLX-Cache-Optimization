#!/usr/bin/env python3
"""Phase 3 entrypoint: learn a contextual-bandit eviction policy and benchmark it vs heuristics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longcache.preflight import preflight

preflight()

from longcache.config import BaselineConfig
from longcache.learned_benchmark import LearnedBenchmark, learned_table


def main():
    config = BaselineConfig()
    benchmark = LearnedBenchmark(config)

    fit = benchmark.setup()
    print("== Fit report ==")
    print(f"model: {fit['model_id']}")
    print(
        f"rollout context: {config.rollout_context} tokens, "
        f"age/future windows: {config.bandit_age}/{config.bandit_future}"
    )
    print(f"eviction context: {config.eviction_context}, budget: {config.eviction_budget}")
    print()

    print("== Collecting rollouts + training policy ==")
    policy = benchmark.train()
    print(f"rows: {policy.rows}")
    print(f"feature coefficients (standardized): {dict(zip(policy.summary()['features'], [round(c, 4) for c in policy.coef]))}")
    print(f"train R²: {policy.train_r2:.4f} | past-attention-only baseline R²: {policy.baseline_r2:.4f}")
    print(f"policy written to {config.policy_path}")
    print()

    results = benchmark.eviction.run(learned_policy=policy)
    results["policy"] = policy.summary()
    from longcache.learned_benchmark import verdict

    results["verdict"] = verdict(results)
    path = benchmark.save(results)

    print(f"== Phase 3 learned vs heuristic eviction @ {results['context_length']} tokens ==")
    print(learned_table(results))
    print()
    print(results["verdict"])
    print()
    print(f"raw results written to {path}")


if __name__ == "__main__":
    main()

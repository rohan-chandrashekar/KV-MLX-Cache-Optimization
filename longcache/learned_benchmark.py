"""Phase 3 orchestration: collect rollouts, train the policy, benchmark it vs the heuristics."""

import json
from pathlib import Path

from .eviction_benchmark import EvictionBenchmark, _fmt
from .learned_eviction import collect_rollouts, train_policy


class LearnedBenchmark:
    def __init__(self, config):
        self.config = config
        self.eviction = EvictionBenchmark(config)

    def setup(self):
        return self.eviction.setup()

    def train(self):
        runtime = self.eviction.runtime
        token_ids = self.eviction.holdout.token_ids(runtime.tokenizer)[
            : self.config.rollout_context
        ]
        features, targets = collect_rollouts(
            runtime, token_ids, self.config.bandit_age, self.config.bandit_future
        )
        policy = train_policy(
            features, targets, runtime, self.config.bandit_age, self.config.bandit_future
        )
        policy.to_json(self.config.policy_path)
        return policy

    def run(self):
        self.setup()
        policy = self.train()
        results = self.eviction.run(learned_policy=policy)
        results["policy"] = policy.summary()
        results["verdict"] = verdict(results)
        return results

    def save(self, results):
        out_dir = Path(self.config.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "phase3_learned.json"
        path.write_text(json.dumps(results, indent=2, default=str))
        return path


def _by_strategy(rows):
    return {row["strategy"]: row for row in rows}


def verdict(results):
    rows = _by_strategy(results["rows"])
    learned = rows.get("learned")
    h2o = rows.get("heavy_hitter")
    if learned is None or h2o is None:
        return "No learned/heavy_hitter rows to compare."
    if learned["perplexity"] is None or h2o["perplexity"] is None:
        return "Perplexity not measured yet (TBD); verdict pending the hardware run."

    ppl_gap = learned["perplexity"] - h2o["perplexity"]
    needle_gap = learned["needle_accuracy"] - h2o["needle_accuracy"]
    speed_ratio = (
        learned["decode_tokens_per_s"] / h2o["decode_tokens_per_s"]
        if h2o["decode_tokens_per_s"]
        else float("nan")
    )
    quality_better = ppl_gap < 0 or needle_gap > 0
    policy = results.get("policy", {})
    lifts = (
        policy.get("train_r2", 0.0) - policy.get("baseline_r2_past_attention_only", 0.0)
    )

    if quality_better:
        head = "Learned eviction BEATS H2O on quality"
    else:
        head = "Learned eviction does NOT beat H2O on quality"
    return (
        f"{head}: perplexity Δ {ppl_gap:+.2f} (lower is better), "
        f"needle Δ {needle_gap:+.2f}, decode speed {speed_ratio:.2f}x vs H2O. "
        f"Reward model lifts R² by {lifts:+.3f} over the past-attention-only baseline "
        f"({policy.get('rows', 0)} rollout rows). "
        "Verdict: the learned policy justifies its cost only if a quality gain offsets the "
        "extra per-step scoring; the numbers above decide it, not intuition."
    )


def learned_table(results):
    header = (
        "| Strategy | KV mem (GB) | Peak mem (GB) | Perplexity | Needle acc. | "
        "Decode tok/s | TTFT (s) |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in results["rows"]:
        lines.append(
            "| {name} | {kv} | {peak} | {ppl} | {needle} | {tps} | {ttft} |".format(
                name=row["strategy"],
                kv=_fmt(row["kv_memory_gb"], ".3f"),
                peak=_fmt(row["peak_memory_gb"], ".2f"),
                ppl=_fmt(row["perplexity"], ".2f"),
                needle=_fmt(row["needle_accuracy"], ".2f"),
                tps=_fmt(row["decode_tokens_per_s"], ".1f"),
                ttft=_fmt(row["ttft_s"], ".2f"),
            )
        )
    return "\n".join(lines)

#!/usr/bin/env python3
"""Phase 5 LinkedIn pack: render charts + a ready-to-paste post draft from real Phase 4 data.

Every number in the draft is computed from results-raw/phase4_master.json — nothing is typed by
hand. Pass an optional machine slug (e.g. M3Pro-18GB) to label the post; otherwise it is detected
from sysctl. Needs only matplotlib + the results file, not MLX.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longcache.charts import headline_stats, load_results, render_all
from longcache.config import BaselineConfig

REPO_URL = "https://github.com/rohan-chandrashekar/KV-MLX-Cache-Optimization"


def detect_slug():
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip().replace("Apple ", "").replace(" ", "")
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
        return f"{chip}-{mem // (1024 ** 3)}GB"
    except Exception:
        return "apple-silicon"


def _pct(value):
    return f"{value:.2f}" if value is not None else "TBD"


def draft_post(stats, slug):
    ctx = stats["reference_context"]
    ratio = f"{stats['best_kv_ratio']:.1f}x" if stats["best_kv_ratio"] else "TBD"
    ppl_delta = (
        f"{stats['best_ppl_delta']:+.2f} perplexity"
        if stats["best_ppl_delta"] is not None
        else "a TBD perplexity change"
    )
    needle = (
        f"{stats['best_needle_accuracy']:.0%} needle-retrieval accuracy"
        if stats["best_needle_accuracy"] is not None
        else "TBD needle accuracy"
    )
    if stats["oom_ceiling"]:
        max_c = stats["max_compressed_context"] or "longer contexts"
        oom_line = (
            f"The FP16 baseline ran out of memory at {stats['oom_ceiling']} tokens, "
            f"while the compressed engine kept going to {max_c} tokens on the same machine."
        )
    else:
        oom_line = "The compressed engine stayed within a bounded memory budget as context grew."
    verdict = stats["learned_verdict"] or "verdict TBD"

    return f"""🧠 LongCache — holding long LLM conversations on-device without running out of memory

On-device LLMs choke on long chats: the KV cache grows with every token and must stay in RAM, so
a long conversation slows down and eventually crashes. I built LongCache on Apple MLX to compress
that cache and measured every trade-off on real Apple Silicon ({slug}).

📊 {stats['model_id']}:
• At {ctx} tokens, {stats['best_method']} cut KV-cache memory {ratio} vs FP16 — for {ppl_delta} and {needle}.
• {oom_line}
• Learned (contextual-bandit) eviction vs the H2O heuristic: {verdict}.

Every number measured on-device, nothing estimated — and where a technique made things worse I
reported the regression honestly. Code, the full comparative table, and the charts below: {REPO_URL}

#OnDeviceAI #AppleSilicon #MLX #LLM #MLSystems #KVCache #EdgeAI
"""


def main():
    config = BaselineConfig()
    results_path = Path(config.results_dir) / "phase4_master.json"
    if not results_path.exists():
        sys.stderr.write(
            f"No benchmark data at {results_path}.\n"
            "Run `python scripts/stress.py` on Apple Silicon first.\n"
        )
        raise SystemExit(1)

    slug = sys.argv[1] if len(sys.argv) > 1 else detect_slug()
    out_dir = Path(__file__).resolve().parent.parent / "charts"

    chart_paths = render_all(results_path, out_dir)
    stats = headline_stats(load_results(results_path))
    post = draft_post(stats, slug)

    post_path = Path(__file__).resolve().parent.parent / f"LINKEDIN-{slug}.md"
    post_path.write_text(post)

    print("== LinkedIn pack ==")
    print(f"machine slug: {slug}")
    print("charts (attach these — they are the post visuals, no screenshots needed):")
    for path in chart_paths:
        print(f"  {path}")
    print(f"draft post written to: {post_path}")
    print()
    print("---- draft post (numbers computed from real data) ----")
    print(post)


if __name__ == "__main__":
    main()

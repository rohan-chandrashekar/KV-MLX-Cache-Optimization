"""Render the two LinkedIn charts from real benchmark data (results-raw/phase4_master.json).

Data extraction is matplotlib-free and unit-testable; plotting imports matplotlib lazily so the
extraction can be validated without it. Nothing is invented: a method with no measured point at
a context (it ran out of memory) simply has no point there, and the OOM ceiling is drawn from
the measured memory budget.
"""

import json
from pathlib import Path

BASELINE = "fp16"
COMPRESSED = ["int8", "int4", "recency", "streaming", "heavy_hitter", "learned"]
METHOD_ORDER = [BASELINE] + COMPRESSED


def load_results(path):
    with open(str(path)) as handle:
        return json.load(handle)


def _ok_points(rows, method, metric):
    points = [
        (row["context_length"], row[metric])
        for row in rows
        if row["method"] == method and row["status"] == "ok" and row.get(metric) is not None
    ]
    return sorted(points)


def _oom_context(rows, method):
    failed = [
        row["context_length"]
        for row in rows
        if row["method"] == method and row["status"] != "ok"
    ]
    return min(failed) if failed else None


def memory_series(results):
    rows = results["rows"]
    series = {m: _ok_points(rows, m, "peak_memory_gb") for m in METHOD_ORDER}
    oom = {m: _oom_context(rows, m) for m in METHOD_ORDER}
    return {m: pts for m, pts in series.items() if pts}, {m: c for m, c in oom.items() if c}


def needle_series(results):
    rows = results["rows"]
    series = {m: _ok_points(rows, m, "needle_accuracy") for m in METHOD_ORDER}
    return {m: pts for m, pts in series.items() if pts}


def _max_context(results):
    contexts = [r["context_length"] for r in results["rows"] if r["status"] == "ok"]
    return max(contexts) if contexts else None


def _baseline_reference_context(results):
    """Largest context where the FP16 baseline actually ran (so a ratio is well-defined even
    when the baseline OOMs at the longest context)."""
    ok = [
        r["context_length"]
        for r in results["rows"]
        if r["method"] == BASELINE and r["status"] == "ok" and r.get("kv_memory_gb")
    ]
    return max(ok) if ok else None


def tradeoff_points(results, context_length=None):
    context_length = context_length or _baseline_reference_context(results) or _max_context(results)
    points = {}
    for row in results["rows"]:
        if (
            row["context_length"] == context_length
            and row["status"] == "ok"
            and row.get("kv_memory_gb") is not None
            and row.get("perplexity") is not None
        ):
            points[row["method"]] = (row["kv_memory_gb"], row["perplexity"])
    return context_length, points


def _rows_at(rows, context_length):
    return {r["method"]: r for r in rows if r["context_length"] == context_length}


def headline_stats(results):
    """Deterministic post-ready numbers, computed from data — never hand-typed.

    The compression ratio is measured at the largest context where the FP16 baseline ran, so it
    stays well-defined even when the baseline OOMs at the longest context. The OOM ceiling and the
    longest context the compressed methods reached are reported separately.
    """
    rows = results["rows"]
    ref_ctx = _baseline_reference_context(results)
    at_ctx = _rows_at(rows, ref_ctx) if ref_ctx else {}
    base = at_ctx.get(BASELINE)
    base_kv = base.get("kv_memory_gb") if base else None
    base_ppl = base.get("perplexity") if base else None

    best_method, best_ratio, best_ppl_delta, best_needle = None, None, None, None
    if base_kv:
        for method in COMPRESSED:
            row = at_ctx.get(method)
            if not row or row["status"] != "ok" or not row.get("kv_memory_gb"):
                continue
            ratio = base_kv / row["kv_memory_gb"]
            if best_ratio is None or ratio > best_ratio:
                best_method, best_ratio = method, ratio
                best_needle = row.get("needle_accuracy")
                best_ppl_delta = (
                    row["perplexity"] - base_ppl
                    if base_ppl is not None and row.get("perplexity") is not None
                    else None
                )

    oom_ceiling = min(
        (r["context_length"] for r in rows if r["method"] == BASELINE and r["status"] != "ok"),
        default=None,
    )
    compressed_ctx = [
        r["context_length"]
        for r in rows
        if r["method"] in COMPRESSED and r["status"] == "ok"
    ]
    max_compressed_ctx = max(compressed_ctx) if compressed_ctx else None

    verdict = None
    both = [
        c
        for c in (r["context_length"] for r in rows)
        if _rows_at(rows, c).get("learned", {}).get("status") == "ok"
        and _rows_at(rows, c).get("heavy_hitter", {}).get("status") == "ok"
    ]
    if both:
        c = max(both)
        learned = _rows_at(rows, c)["learned"]
        h2o = _rows_at(rows, c)["heavy_hitter"]
        if learned.get("perplexity") is not None and h2o.get("perplexity") is not None:
            d = learned["perplexity"] - h2o["perplexity"]
            verdict = (
                f"learned beats H2O (perplexity {d:+.2f} at {c} tokens)"
                if d < 0
                else f"learned does not beat H2O (perplexity {d:+.2f} at {c} tokens)"
            )

    return {
        "reference_context": ref_ctx,
        "context_length": ref_ctx,
        "max_compressed_context": max_compressed_ctx,
        "model_id": results.get("model_id"),
        "budget": results.get("budget"),
        "baseline_peak_gb": base.get("peak_memory_gb") if base else None,
        "baseline_kv_gb": base_kv,
        "best_method": best_method,
        "best_kv_ratio": best_ratio,
        "best_ppl_delta": best_ppl_delta,
        "best_needle_accuracy": best_needle,
        "oom_ceiling": oom_ceiling,
        "learned_verdict": verdict,
    }


def plot_memory_vs_context(results, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, oom = memory_series(results)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        if method not in series:
            continue
        xs, ys = zip(*series[method])
        width = 2.6 if method == BASELINE else 1.6
        ax.plot(xs, ys, marker="o", linewidth=width, label=method)

    budget = results.get("budget_gb")
    if budget:
        ax.axhline(budget, color="crimson", linestyle="--", linewidth=1.5)
        ax.text(
            ax.get_xlim()[1], budget, "  memory budget (OOM ceiling)",
            color="crimson", va="center", ha="left", fontsize=9,
        )
    for method, ctx in oom.items():
        if budget:
            ax.scatter([ctx], [budget], color="crimson", marker="x", s=80, zorder=5)

    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Peak unified memory (GB)")
    ax.set_title("KV-cache compression keeps on-device memory bounded as context grows")
    ax.grid(True, alpha=0.3)
    ax.legend(title="method", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


def plot_needle_vs_context(results, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = needle_series(results)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        if method not in series:
            continue
        xs, ys = zip(*series[method])
        width = 2.6 if method == BASELINE else 1.6
        ax.plot(xs, ys, marker="o", linewidth=width, label=method)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Needle-in-a-haystack accuracy")
    ax.set_title("Retrieval accuracy vs context length under each compression method")
    ax.grid(True, alpha=0.3)
    ax.legend(title="method", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


def plot_tradeoff(results, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context_length, points = tradeoff_points(results)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, (kv, ppl) in points.items():
        is_base = method == BASELINE
        ax.scatter([kv], [ppl], s=130 if is_base else 90, zorder=4,
                   marker="*" if is_base else "o")
        ax.annotate(method, (kv, ppl), textcoords="offset points", xytext=(6, 5), fontsize=9)

    ax.set_xlabel("KV-cache memory (GB) — lower is better →")
    ax.invert_xaxis()
    ax.set_ylabel("Perplexity — lower is better ↓")
    ax.set_title(
        f"Memory vs quality trade-off at {context_length} tokens (best = bottom-right)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


def render_all(results_path, out_dir):
    results = load_results(results_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_memory_vs_context(results, out_dir / "memory_vs_context.png"),
        plot_needle_vs_context(results, out_dir / "needle_vs_context.png"),
        plot_tradeoff(results, out_dir / "memory_quality_tradeoff.png"),
    ]

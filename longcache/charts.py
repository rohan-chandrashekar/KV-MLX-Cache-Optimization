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


def render_all(results_path, out_dir):
    results = load_results(results_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_memory_vs_context(results, out_dir / "memory_vs_context.png"),
        plot_needle_vs_context(results, out_dir / "needle_vs_context.png"),
    ]

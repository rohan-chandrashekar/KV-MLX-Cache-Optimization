#!/usr/bin/env python3
"""Phase 5 entrypoint: render the memory-vs-context and needle-vs-context charts.

Reads real benchmark data (results-raw/phase4_master.json) and writes PNGs to charts/. Needs
only matplotlib + the results file, not MLX — but it refuses to invent data: run the Phase 4
stress benchmark first so the charts render from measured numbers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from longcache.charts import render_all
from longcache.config import BaselineConfig


def main():
    config = BaselineConfig()
    results_path = Path(config.results_dir) / "phase4_master.json"
    if not results_path.exists():
        sys.stderr.write(
            f"No benchmark data at {results_path}.\n"
            "Run `python scripts/stress.py` on Apple Silicon first; charts render from real "
            "measured numbers, never placeholders.\n"
        )
        raise SystemExit(1)

    out_dir = Path(__file__).resolve().parent.parent / "charts"
    paths = render_all(results_path, out_dir)
    print("Wrote charts:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

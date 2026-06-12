# Prompt: Run LongCache on this Apple Silicon Mac and capture the numbers

Copy everything in the fenced block below and paste it into Claude Code running on the
Apple Silicon Mac. It detects exactly which M-series machine it is, runs the whole project,
and records every measured number tagged to that machine. Safe to run on several different
Macs — each writes its own results file keyed by chip + RAM, so nothing collides.

---

```
You are Claude Code on an Apple Silicon Mac. Run the LongCache KV-cache-compression project
end-to-end, detect exactly which Mac this is, and record every measured number tagged to THIS
machine.

CARDINAL RULE (non-negotiable): every number must be genuinely measured on this machine. Never
fabricate, estimate, extrapolate, or round up. If something cannot run or fails, write down what
happened and move on — a clearly explained failure or an OOM is valid data, a fake number is not.
Paste tables exactly as the scripts print them; do not hand-edit a single digit.

STEP 0 — Detect this machine (do this first, record output verbatim):
  sw_vers
  uname -m                                   # must be arm64
  sysctl -n machdep.cpu.brand_string         # e.g. "Apple M3 Pro"
  system_profiler SPHardwareDataType | grep -E "Model Name|Chip|Memory|Total Number of Cores"
  python3 --version                          # must be >= 3.10
Build a machine SLUG from chip + RAM, lower-friction form, e.g.:
  CHIP=$(sysctl -n machdep.cpu.brand_string | sed 's/Apple //; s/ //g')   # -> M3Pro
  RAM=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))                 # -> 18
  SLUG="${CHIP}-${RAM}GB"                                                  # -> M3Pro-18GB
Use $SLUG in the results filename and in every table caption. If you ever run two machines with
the same chip+RAM, append the Model Name or a date so the slug stays unique.

STEP 1 — Get the code (skip the clone if you are already inside the repo):
  git clone https://github.com/rohan-chandrashekar/KV-MLX-Cache-Optimization.git
  cd KV-MLX-Cache-Optimization
Then read CLAUDE.md and PROGRESS.md before doing anything else.

STEP 2 — Environment (the first model load downloads ~1 GB from Hugging Face; needs network):
  python3 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
Record the exact stack for reproducibility: pip show mlx mlx-lm scikit-learn | grep -E "Name|Version"

STEP 3 — Data + smoke test (this is the go/no-go gate):
  python scripts/get_data.py
  python scripts/smoke.py
If smoke.py does NOT end with "Smoke test PASSED", STOP. Paste the failing check(s) and the
traceback, say which path broke, and do not run the benchmarks. (smoke.py exercises every code
path with tiny inputs in ~30s, so a real integration problem shows up here instead of 20 minutes
into a benchmark.)

STEP 4 — Run the phases and capture every table (only after smoke passes):
  python scripts/baseline.py   2>&1 | tee /tmp/lc_p0.txt    # baseline + telemetry + OOM ceiling
  python scripts/quantize.py   2>&1 | tee /tmp/lc_p1.txt    # FP16 vs INT8 vs INT4
  python scripts/evict.py      2>&1 | tee /tmp/lc_p2.txt    # recency / StreamingLLM / H2O
  python scripts/learn.py      2>&1 | tee /tmp/lc_p3.txt    # learned policy vs heuristics + verdict
  python scripts/stress.py     2>&1 | tee /tmp/lc_p4.txt    # master comparative table (the big run)
  python scripts/charts.py                                  # renders charts/*.png from Phase 4 data
Expectations and honest handling:
  - evict.py / learn.py / stress.py prefill token-by-token for H2O and the learned policy, so they
    are SLOW at 16k (minutes per context). High TTFT for those methods is expected and is itself a
    reported result — not a bug.
  - On a tight-RAM Mac (8 GB), some methods/contexts will hit OOM. The harness records "OOM/fail"
    and keeps going; that is the point of the experiment (baseline hits its ceiling, compressed
    methods stay bounded). Note the OOM ceiling rather than treating it as failure.
  - If a phase is taking unreasonably long just to get a first pass, you MAY temporarily lower
    context_lengths (and/or eviction_context, rollout_context, needle_depths) in
    longcache/config.py — but then say so explicitly and record the exact values you used.
  - If the model does not fit at all (assert_fits aborts), report it and try the run with smaller
    context_lengths; do not silently skip.

STEP 5 — Record the numbers, tagged to THIS machine. Create RESULTS-$SLUG.md containing:
  - A "Machine" header: Model Name, Chip, RAM, core counts, macOS version, Python version,
    mlx + mlx-lm + scikit-learn versions, the model id, and whether it fit the memory budget.
  - One section per phase with the printed markdown table copied VERBATIM (Phase 0 baseline +
    OOM ceiling; Phase 1 quantization; Phase 2 eviction; Phase 3 learned table + the verdict line;
    Phase 4 master comparative table).
  - The two chart paths (charts/memory_vs_context.png, charts/needle_vs_context.png).
  - Any config changes you made and any failures/OOMs, stated plainly.

STEP 6 — (Optional) commit, only if this Mac has git push access:
  git add RESULTS-$SLUG.md && git commit -m "Measured results on $SLUG"
  git push -u origin HEAD
The filename is unique per machine, so parallel runs on different Macs do not collide. Do NOT
commit .venv, model weights, data/holdout.txt, results-raw/, or charts/ — those are gitignored.
If you want the charts in the repo, git add them explicitly (they are small) and say so.

STEP 7 — Report back a short summary: the machine ($SLUG and full chip name), whether smoke
passed, which phases completed, and the headline numbers — baseline sustained tok/s and the OOM
ceiling, INT4 KV-memory cut and its perplexity delta, the best eviction strategy at 16k, and the
learned-vs-H2O verdict — plus the path to RESULTS-$SLUG.md and the two PNGs. Keep it factual; cite
only what the scripts actually printed.
```

---

## Notes for you (the human)

- **Run it on each Mac you have access to.** Every machine writes its own `RESULTS-<chip>-<RAM>GB.md`
  (e.g. `RESULTS-M2-16GB.md`, `RESULTS-M3Pro-18GB.md`), so you end up with a clean per-machine
  record and can compare an M1 against an M3 Max directly.
- **`smoke.py` is the safety net.** If it passes, the long benchmarks will run; if it fails, the
  agent stops and shows you the exact broken check in seconds.
- **Send the numbers back here.** Paste the contents of any `RESULTS-<slug>.md` (and the two PNGs)
  into the original Claude project chat and I'll fill the `TBD`s in README / RESUME_BULLETS /
  PROGRESS with the real measured values — choosing one machine as the canonical README result and
  keeping the rest as a cross-device comparison.
- **Low-RAM Macs are still useful.** An 8 GB machine that OOMs the FP16 baseline early but runs the
  compressed methods is exactly the story the project is about — that contrast is a feature, not a
  failed run.

# LongCache — On-Device Long-Context LLM via KV-Cache Compression

Flagship portfolio project for an Apple AIML internship application. Must survive a senior Apple ML systems engineer's scrutiny in a technical interview.

## Cardinal rule
Every performance number must be genuinely measured on this machine. Never fabricate, estimate, extrapolate, or round up a metric. If a number cannot be measured yet, write "TBD" and state why. If a technique makes things worse, report the regression honestly — a clearly explained negative result is a valid outcome. A wrong-but-impressive number is a failure; a modest-but-real number is a success.

## Where we are
- Current state, completed phases, known issues: read `PROGRESS.md`.
- Full phase-by-phase build plan: read `BUILD_PROMPT.md`.
- At the start of a session, read both before doing anything else, then continue from the phase the user names.

## Workflow rules
- Build ONE phase at a time. Do not skip ahead.
- At the end of each phase: run the verification, update `README.md`, `RESUME_BULLETS.md`, and `PROGRESS.md` with real measured results, commit, then STOP, summarize for the user, and wait for "go".
- Commit and push after every meaningful step. This is a wiped lab machine; uncommitted work is lost on logout and auto memory does not persist here. `CLAUDE.md` plus `PROGRESS.md` in the repo are the only durable memory.
- Before ending a session, remind the user to commit and push.

## Tech stack and constraints
- Apple Silicon, macOS 14+.
- Apple MLX and mlx-lm (Python). Use mlx-lm's model loading, generation, and KV cache abstractions (quantized and rotating/windowed cache classes) — verify the current API rather than assuming method names.
- Model: a SMALL 4-bit model (1–3B params) that fits the machine's unified memory and supports long context. Confirm it fits before generating. No 8B models on a shared lab machine.

## Hard "do not" list
- NO runtime SVD or low-rank decomposition in the decode loop — it costs more than the attention it would save. Compress by quantization and eviction only.
- Quality is measured OFFLINE: perplexity on held-out long text + needle-in-a-haystack retrieval accuracy. NEVER "bound perplexity" during live generation — perplexity needs reference text.
- `powermetrics` utilization needs sudo, which this machine may not allow. Report memory, throughput, and timing, which do not.

## Coding standards
- Complete, runnable files. No inline code comments; clear names, explanation in commit messages and README prose. Modular, object-oriented where it helps.
- Clearly flag placeholder values for anything not provided.
- Communicate directly. Push back on anything wrong, slow, or that won't survive interview scrutiny rather than agreeing.

## Never commit
Model weights, training rollouts, logs, and any large benchmark artifacts.

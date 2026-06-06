You are building a macOS portfolio project called "LongCache," an on-device long-context
LLM inference engine that compresses the key-value (KV) cache. It is a flagship project for
my Apple internship application (AIML track). It must survive scrutiny from a senior Apple
ML systems engineer in a technical interview.

THE NON-NEGOTIABLE RULE: every performance number must be genuinely measured on this
machine. Never fabricate, estimate, extrapolate, or round up a metric. If a number cannot
be measured yet, write "TBD" and state why. A wrong-but-impressive number is a failure; a
modest-but-real number is a success. If a technique makes things worse (e.g. adds latency),
report the regression honestly -- a negative result, clearly explained, is a valid outcome.

WHAT IT DOES
Keeps long conversations alive on a fixed memory budget by compressing the LLM's KV cache,
on Apple Silicon with MLX. It quantizes the cache, evicts low-value tokens with heuristics,
and tests whether a learned (contextual-bandit) eviction policy beats those heuristics.

THE PROBLEM (anchor every decision to this)
On-device LLMs choke on long conversations because the KV cache grows linearly with every
token and must stay in RAM. On fixed unified memory, a long chat slows generation and
eventually crashes with an out-of-memory error, and offloading to the cloud destroys the
privacy that is the reason to run on-device. The goal: hold the longest possible
conversation, as fast as possible, on a fixed memory budget, while losing as little model
quality as possible. The hard part is the tradeoff -- evict old tokens and the model forgets;
keep everything and it crashes.

HARD CONSTRAINTS
- Apple Silicon, macOS 14+. Confirm with `sw_vers`, `uname -m`.
- Framework: Apple MLX and mlx-lm (Python). Use mlx-lm's model loading, generation, and KV
  cache abstractions (it provides quantized and rotating/windowed cache classes -- verify the
  current API rather than assuming method names).
- Model: a SMALL 4-bit model (1-3B parameters, e.g. a Llama-3.2 or Qwen2.5 small instruct
  model from mlx-community) that fits the machine's unified memory and supports long context.
  Confirm it fits before generating. Do NOT use an 8B model on a shared lab machine.
- NO runtime SVD or low-rank decomposition in the decode loop -- it costs more than the
  attention it would save. Compression is by quantization and eviction only.
- Quality is measured OFFLINE, two ways: perplexity on held-out long text, and
  needle-in-a-haystack retrieval accuracy vs context length. NEVER attempt to "bound
  perplexity" during live generation -- perplexity needs reference text and cannot be
  computed on free generation.
- Utilization metrics from `powermetrics` need sudo, which this managed lab machine may not
  allow. Report what you can measure without sudo: peak/KV memory, throughput, and timing.

ENGINEERING STANDARDS
- Complete, runnable files. No inline code comments; clear names, explanation in commit
  messages and README prose. Modular, object-oriented where it helps.
- Clearly flag placeholder values for anything not provided.
- Communicate directly; push back on anything wrong, slow, or that won't survive interview
  scrutiny rather than agreeing.
- Git: one commit per completed phase. Maintain a .gitignore excluding the venv, model
  weights, training rollouts, and logs.
- Maintain three living docs every phase: README.md (numbers-first, real measured values),
  RESUME_BULLETS.md (interview-defensible bullets, measured numbers only), PROGRESS.md
  (done / next / known issues).

ITERATIVE PROTOCOL (IMPORTANT)
Build ONE phase at a time; do not skip ahead. At the end of each phase you MUST:
(1) run the verification, (2) update README.md, RESUME_BULLETS.md, and PROGRESS.md with the
real results, (3) commit, (4) STOP and summarize -- what you built, the measured numbers,
what didn't work, what the next phase does -- then WAIT for me to reply "go." Only ask when
genuinely blocked; otherwise make reasonable defaults and note them.

PHASES

Phase 0 -- Baseline long-context loop + telemetry
Initialize MLX/mlx-lm, load a small 4-bit model, build a deterministic generation loop, and
build the measurement harness: Time-to-First-Token, sustained tokens/sec, peak unified
memory, KV-cache memory as context grows, perplexity on a held-out long-text set, and a
needle-in-a-haystack retrieval accuracy test across context lengths.
Done when: a baseline table (TTFT, tok/s, peak + KV memory, perplexity, needle accuracy)
across context lengths, and the context length where memory becomes the binding constraint
(the OOM ceiling) on this machine.

Phase 1 -- KV-cache quantization
Store cached keys and values in INT8 and INT4 (use mlx-lm's quantized cache if available,
else implement). Keep all tokens; each costs less memory.
Done when: a before/after table at several context lengths -- memory, perplexity delta,
needle accuracy, tokens/sec -- versus the FP16 baseline.

Phase 2 -- Heuristic eviction
Implement eviction that keeps fewer tokens: a recency window, attention-sink + recent
(StreamingLLM style), and heavy-hitter eviction (keep the highest-attention tokens, H2O
style). Use mlx-lm's rotating/windowed cache for the recency part; add heavy-hitter on top.
Done when: a table comparing the eviction strategies on memory, perplexity delta, needle
accuracy, and tokens/sec at 16k context.

Phase 3 -- Learned (contextual-bandit) eviction
Frame each keep/evict decision as a contextual bandit: features per token (normalized
attention received, recency/position, value norm, layer), action = keep or evict, reward =
quality preserved minus memory cost. Collect rollouts by running the model, train a
lightweight policy (scikit-learn or a small MLX model), then benchmark it head-to-head
against the Phase 2 heuristics. Report the truth: if the learned policy does not beat the
heuristic, say so with the data and explain why.
Done when: a comparative table (learned vs heuristic eviction) on memory, perplexity delta,
needle accuracy, and tokens/sec, with an explicit verdict on whether the learned policy
justifies its cost.

Phase 4 -- Stress benchmark + master comparative table
Push context to 16k+ (within what the model and machine allow) and produce one comparative
markdown table: baseline vs quantized vs heuristic-evicted vs learned-evicted -- peak
memory, KV memory, tokens/sec, TTFT, perplexity delta, needle accuracy, at each context
length.
Done when: the master comparative table is in README.md with real numbers.

Phase 5 -- Docs + bullets + chart
Finalize README, RESUME_BULLETS.md, and a DEMO.md. Produce a memory-vs-context chart
(baseline climbing to OOM vs the compressed engine staying bounded) plus a needle-accuracy
vs context chart for a LinkedIn post.
Done when: docs finalized and the charts render from real benchmark data.

Start now with Phase 0: confirm hardware, set up the repo and .gitignore, pick and load a
small 4-bit model that fits this machine, build the baseline loop and telemetry, measure it,
and stop.

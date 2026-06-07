"""Heavy-hitter (H2O-style) KV eviction.

mlx-lm runs attention through the fused mx.fast.scaled_dot_product_attention, which never
materializes the attention weights heavy-hitter eviction needs to score tokens. To recover
those weights we scope-patch the model module's `scaled_dot_product_attention` with an
explicit-softmax equivalent that returns the same output and, when the layer cache is a
HeavyHitterCache, accumulates per-key attention mass into it.

The policy is driven one token at a time. A single-query step makes create_attention_mask
return None (so no mask alignment problem when the cache holds a non-contiguous subset of past
positions), keeps the score matrix to shape [heads, 1, retained] instead of the quadratic
[heads, L, L] a long-context explicit forward would need, and lets evicted survivors keep
their original absolute RoPE: the cache reports total-processed as its offset, so each new
token is roped at its true position. The cost is that prefill is un-fused and runs token by
token, which is the honest runtime price of scoring attention the fused kernel hides.
"""

import importlib
import math
import time
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

from .generation import _eos_ids
from .telemetry import MemoryProbe, kv_cache_gb


class HeavyHitterCache:
    def __init__(self, budget, sink=4, recent=1024):
        if budget < sink + recent:
            raise ValueError(
                f"budget {budget} must be >= sink {sink} + recent {recent}."
            )
        self.budget = budget
        self.sink = sink
        self.recent = recent
        self.keys = None
        self.values = None
        self.scores = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        new = keys.shape[2]
        self.offset += new
        if self.keys is None:
            self.keys = keys
            self.values = values
            self.scores = mx.zeros((new,), dtype=mx.float32)
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
            self.scores = mx.concatenate(
                [self.scores, mx.zeros((new,), dtype=mx.float32)], axis=0
            )
        self._evict()
        return self.keys, self.values

    def _evict(self):
        size = self.keys.shape[2]
        if size <= self.budget:
            return
        n_drop = size - self.budget
        mid_start = self.sink
        mid_end = size - self.recent
        middle = self.scores[mid_start:mid_end]
        order = mx.argsort(middle)
        keep_local = mx.sort(order[n_drop:])
        head = mx.arange(mid_start).astype(mx.int32)
        kept = (keep_local + mid_start).astype(mx.int32)
        tail = mx.arange(mid_end, size).astype(mx.int32)
        keep_idx = mx.concatenate([head, kept, tail])
        self.keys = mx.take(self.keys, keep_idx, axis=2)
        self.values = mx.take(self.values, keep_idx, axis=2)
        self.scores = mx.take(self.scores, keep_idx, axis=0)

    def record_scores(self, column):
        if self.scores is None:
            return
        if column.shape[0] != self.scores.shape[0]:
            return
        self.scores = self.scores + column

    @property
    def state(self):
        return self.keys, self.values


def _explicit_attention(queries, keys, values, cache, scale, mask, sinks=None):
    n_heads = queries.shape[1]
    n_kv = keys.shape[1]
    if n_heads != n_kv:
        repeats = n_heads // n_kv
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)
    scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
    if mask is not None:
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -1e9)
        else:
            scores = scores + mask
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(values.dtype)
    if hasattr(cache, "record_scores"):
        cache.record_scores(weights.sum(axis=(0, 1, 2)).astype(mx.float32))
    return weights @ values


@contextmanager
def capture_scores(model):
    module_name = type(model).__module__
    module = importlib.import_module(module_name)
    if not hasattr(module, "scaled_dot_product_attention"):
        raise RuntimeError(
            f"{module_name} does not import scaled_dot_product_attention as a bare name; "
            "heavy-hitter capture cannot patch this architecture."
        )
    original = module.scaled_dot_product_attention
    module.scaled_dot_product_attention = _explicit_attention
    try:
        yield
    finally:
        module.scaled_dot_product_attention = original


class TokenStreamRunner:
    """Token-by-token prefill/decode under attention capture, for any streaming-eviction cache.

    Single-query steps keep create_attention_mask at None and bound the captured score matrix;
    every eviction cache (H2O, learned) plugs in through the make_cache factory.
    """

    def __init__(self, runtime, make_cache):
        self.runtime = runtime
        self._make_cache = make_cache
        self.memory = MemoryProbe()

    def fresh_cache(self):
        return self._make_cache()

    def _prefill(self, model, cache, prompt_ids):
        logits = None
        for i in range(int(prompt_ids.shape[0])):
            logits = model(prompt_ids[i : i + 1][None], cache=cache)
            mx.eval(logits)
        return logits

    def generate(self, prompt_ids, max_tokens, stop_on_eos=True):
        model = self.runtime.model
        eos = _eos_ids(self.runtime.tokenizer) if stop_on_eos else set()
        cache = self.fresh_cache()
        self.memory.reset()

        tokens = []
        with capture_scores(model):
            start = time.perf_counter()
            logits = self._prefill(model, cache, prompt_ids)
            ttft = time.perf_counter() - start
            first_token_time = time.perf_counter()
            last = logits[:, -1, :]
            for _ in range(max_tokens):
                next_id = mx.argmax(last, axis=-1)
                mx.eval(next_id)
                token_id = int(next_id.item())
                tokens.append(token_id)
                if token_id in eos:
                    break
                logits = model(next_id[None], cache=cache)
                last = logits[:, -1, :]
            end = time.perf_counter()

        decoded = len(tokens)
        if decoded > 1:
            decode_tps = (decoded - 1) / (end - first_token_time)
        else:
            decode_tps = 0.0
        return {
            "prompt_tokens": int(prompt_ids.shape[0]),
            "decoded_tokens": decoded,
            "ttft_s": ttft,
            "decode_tokens_per_s": decode_tps,
            "peak_memory_gb": self.memory.peak_gb(),
            "kv_memory_gb": kv_cache_gb(cache),
            "token_ids": tokens,
            "text": self.runtime.tokenizer.decode(tokens),
        }

    def perplexity(self, token_ids):
        model = self.runtime.model
        ids = mx.array(token_ids)
        length = int(ids.shape[0])
        if length < 2:
            raise ValueError("Need at least two tokens to compute perplexity.")
        cache = self.fresh_cache()
        nll_sum = 0.0
        counted = 0
        with capture_scores(model):
            for i in range(length - 1):
                logits = model(ids[i : i + 1][None], cache=cache)[:, -1, :]
                ce = nn.losses.cross_entropy(logits, ids[i + 1 : i + 2], reduction="none")
                nll_sum += float(ce.sum().item())
                counted += 1
        mean_nll = nll_sum / counted
        return {
            "perplexity": math.exp(mean_nll),
            "mean_nll": mean_nll,
            "tokens_scored": counted,
        }


class HeavyHitterRunner(TokenStreamRunner):
    def __init__(self, runtime, budget, sink, recent):
        self.budget = budget
        self.sink = sink
        self.recent = recent
        num_layers = len(runtime.model.layers)
        super().__init__(
            runtime,
            lambda: [HeavyHitterCache(budget, sink, recent) for _ in range(num_layers)],
        )

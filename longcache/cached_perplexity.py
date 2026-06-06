"""Teacher-forced perplexity computed through a live (optionally quantized) KV cache.

Running the forward pass with a quantized cache makes attention route through
quantized_scaled_dot_product_attention, so the resulting perplexity reflects the real
quality cost of KV quantization rather than a clean uncached forward.
"""

import math

import mlx.core as mx
import mlx.nn as nn


def cached_perplexity(model, token_ids, cache, chunk_size):
    seq_len = len(token_ids)
    if seq_len < 2:
        raise ValueError("Need at least two tokens to compute perplexity.")
    ids = mx.array(token_ids)

    nll_sum = 0.0
    counted = 0
    pos = 0
    while pos < seq_len:
        end = min(pos + chunk_size, seq_len)
        chunk = ids[pos:end]
        logits = model(chunk[None], cache=cache)[0]
        n_pred = min(end - pos, seq_len - 1 - pos)
        if n_pred > 0:
            targets = ids[pos + 1 : pos + 1 + n_pred]
            per_token = nn.losses.cross_entropy(
                logits[:n_pred], targets, reduction="none"
            )
            nll_sum += float(per_token.sum().item())
            counted += n_pred
        pos = end

    mean_nll = nll_sum / counted
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "tokens_scored": counted,
        "chunk_size": chunk_size,
    }

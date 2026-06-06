"""Offline sliding-window perplexity on held-out long text."""

import math

import mlx.core as mx
import mlx.nn as nn


def sliding_window_perplexity(model, token_ids, max_len, stride):
    seq_len = len(token_ids)
    if seq_len < 2:
        raise ValueError("Need at least two tokens to compute perplexity.")
    ids = mx.array(token_ids)

    nll_sum = 0.0
    counted = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        window = ids[begin:end]
        logits = model(window[None])[0]
        per_token = nn.losses.cross_entropy(
            logits[:-1], window[1:], reduction="none"
        )
        target_len = end - prev_end
        take = min(target_len, per_token.shape[0])
        chunk = per_token[-take:]
        nll_sum += float(chunk.sum().item())
        counted += take
        prev_end = end
        if end == seq_len:
            break

    mean_nll = nll_sum / counted
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "tokens_scored": counted,
        "max_len": max_len,
        "stride": stride,
    }

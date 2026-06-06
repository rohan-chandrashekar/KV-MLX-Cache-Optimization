"""Deterministic generation loop with TTFT and sustained-throughput measurement."""

import time

import mlx.core as mx

try:
    from mlx_lm.generate import generate_step
except ImportError:
    from mlx_lm.utils import generate_step

from .telemetry import MemoryProbe, kv_cache_gb


def argmax_sampler(logits):
    return mx.argmax(logits, axis=-1)


def _eos_ids(tokenizer):
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(ids)
    single = getattr(tokenizer, "eos_token_id", None)
    return {single} if single is not None else set()


class GenerationRun:
    def __init__(self, runtime):
        self.runtime = runtime
        self.memory = MemoryProbe()

    def run(
        self,
        prompt_ids,
        max_tokens,
        prompt_cache=None,
        stop_on_eos=True,
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
    ):
        model = self.runtime.model
        tokenizer = self.runtime.tokenizer
        eos = _eos_ids(tokenizer) if stop_on_eos else set()
        if prompt_cache is None:
            prompt_cache = self.runtime.new_cache()

        prompt_len = int(prompt_ids.shape[0])
        self.memory.reset()

        tokens = []
        ttft = None
        first_token_time = None
        start = time.perf_counter()

        stepper = generate_step(
            prompt_ids,
            model,
            max_tokens=max_tokens,
            sampler=argmax_sampler,
            prompt_cache=prompt_cache,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        for index, (token, _logprobs) in enumerate(stepper):
            mx.eval(token)
            now = time.perf_counter()
            if index == 0:
                ttft = now - start
                first_token_time = now
            token_id = int(token.item())
            tokens.append(token_id)
            if token_id in eos:
                break
        end = time.perf_counter()

        decoded = len(tokens)
        if decoded > 1:
            decode_tps = (decoded - 1) / (end - first_token_time)
        else:
            decode_tps = 0.0

        return {
            "prompt_tokens": prompt_len,
            "decoded_tokens": decoded,
            "kv_bits": kv_bits,
            "ttft_s": ttft,
            "decode_tokens_per_s": decode_tps,
            "peak_memory_gb": self.memory.peak_gb(),
            "kv_memory_gb": kv_cache_gb(prompt_cache),
            "token_ids": tokens,
        }

    def run_text(self, prompt_ids, max_tokens, **kwargs):
        result = self.run(prompt_ids, max_tokens, **kwargs)
        result["text"] = self.runtime.tokenizer.decode(result["token_ids"])
        return result

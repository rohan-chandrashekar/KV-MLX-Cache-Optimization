"""Timing and unified-memory probes built on the MLX memory API (bytes)."""

import time

import mlx.core as mx

BYTES_PER_GB = 1024 ** 3


class Stopwatch:
    def __init__(self):
        self._start = None
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


class MemoryProbe:
    def reset(self):
        mx.reset_peak_memory()

    def active_gb(self):
        return mx.get_active_memory() / BYTES_PER_GB

    def peak_gb(self):
        return mx.get_peak_memory() / BYTES_PER_GB


def array_bytes(obj):
    if obj is None:
        return 0
    if isinstance(obj, (tuple, list)):
        return sum(array_bytes(o) for o in obj)
    nbytes = getattr(obj, "nbytes", None)
    return int(nbytes) if nbytes is not None else 0


def kv_cache_bytes(prompt_cache):
    return sum(array_bytes(layer.state) for layer in prompt_cache)


def kv_cache_gb(prompt_cache):
    return kv_cache_bytes(prompt_cache) / BYTES_PER_GB

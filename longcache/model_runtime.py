"""Model loading, config introspection, fit checks, and KV-cache construction."""

import subprocess
from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from .telemetry import BYTES_PER_GB

FP16_BYTES = 2


def unified_memory_bytes():
    out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
    return int(out)


@dataclass
class ModelDims:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    num_heads: int


def _first_attr(obj, names, default=None):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def model_dims(model):
    args = getattr(model, "args", model)
    hidden = _first_attr(args, ["hidden_size", "model_dim", "d_model"])
    num_heads = _first_attr(args, ["num_attention_heads", "n_heads"])
    num_kv = _first_attr(
        args, ["num_key_value_heads", "n_kv_heads", "num_attention_heads", "n_heads"]
    )
    num_layers = _first_attr(args, ["num_hidden_layers", "n_layers", "num_layers"])
    head_dim = _first_attr(args, ["head_dim"])
    if head_dim is None and hidden is not None and num_heads:
        head_dim = hidden // num_heads
    return ModelDims(
        num_layers=int(num_layers),
        num_kv_heads=int(num_kv),
        head_dim=int(head_dim),
        hidden_size=int(hidden),
        num_heads=int(num_heads),
    )


def analytical_kv_bytes(dims, seq_len, bytes_per_elem=FP16_BYTES):
    return 2 * dims.num_layers * dims.num_kv_heads * dims.head_dim * seq_len * bytes_per_elem


def parameter_bytes(model):
    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


class ModelRuntime:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dims = None

    def load(self):
        mx.random.seed(self.config.seed)
        self.model, self.tokenizer = load(self.config.model_id)
        self.dims = model_dims(self.model)
        return self

    def weight_gb(self):
        return parameter_bytes(self.model) / BYTES_PER_GB

    def budget_gb(self):
        return unified_memory_bytes() / BYTES_PER_GB * self.config.memory_budget_fraction

    def projected_gb(self, seq_len):
        kv = analytical_kv_bytes(self.dims, seq_len) / BYTES_PER_GB
        return self.weight_gb() + kv

    def fit_report(self):
        max_ctx = max(self.config.context_lengths)
        return {
            "model_id": self.config.model_id,
            "weight_gb": self.weight_gb(),
            "unified_memory_gb": unified_memory_bytes() / BYTES_PER_GB,
            "budget_gb": self.budget_gb(),
            "analytical_kv_gb_at_max_ctx": analytical_kv_bytes(self.dims, max_ctx)
            / BYTES_PER_GB,
            "projected_gb_at_max_ctx": self.projected_gb(max_ctx),
            "dims": self.dims,
        }

    def assert_fits(self):
        report = self.fit_report()
        if report["projected_gb_at_max_ctx"] > report["budget_gb"]:
            raise RuntimeError(
                "Model does not fit the memory budget: projected "
                f"{report['projected_gb_at_max_ctx']:.2f} GB at "
                f"{max(self.config.context_lengths)} tokens exceeds budget "
                f"{report['budget_gb']:.2f} GB. Pick a smaller model or shorter context."
            )
        return report

    def new_cache(self, max_kv_size=None):
        return make_prompt_cache(self.model, max_kv_size=max_kv_size)

    def quantized_cache(self, kv_bits, group_size=64):
        cache = make_prompt_cache(self.model)
        if kv_bits is None:
            return cache
        quantized = []
        for layer in cache:
            if hasattr(layer, "to_quantized"):
                quantized.append(layer.to_quantized(group_size=group_size, bits=kv_bits))
            else:
                quantized.append(layer)
        return quantized

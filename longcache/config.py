"""Experiment configuration for the Phase 0 baseline sweep."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BaselineConfig:
    model_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    seed: int = 0

    context_lengths: List[int] = field(
        default_factory=lambda: [2048, 4096, 8192, 16384]
    )
    decode_tokens: int = 64

    perplexity_max_len: int = 4096
    perplexity_stride: int = 2048
    perplexity_token_budget: int = 65536

    needle_depths: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    needle_answer_tokens: int = 16

    kv_precisions: List = field(default_factory=lambda: ["fp16", 8, 4])
    kv_group_size: int = 64
    quant_chunk_size: int = 256

    eviction_context: int = 16384
    eviction_budget: int = 2048
    streaming_sink: int = 4
    heavy_sink: int = 4
    heavy_recent: int = 1024

    memory_budget_fraction: float = 0.80
    oom_probe_step: int = 8192
    oom_probe_max: int = 131072

    holdout_path: Path = REPO_ROOT / "data" / "holdout.txt"
    results_dir: Path = REPO_ROOT / "results-raw"

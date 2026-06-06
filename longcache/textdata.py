"""Held-out text loading and tokenization helpers for the offline quality tests."""

from pathlib import Path

import mlx.core as mx


class HoldoutText:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Held-out text not found at {self.path}. "
                "Run `python scripts/get_data.py` first."
            )
        self.text = self.path.read_text(encoding="utf-8", errors="ignore")

    def token_ids(self, tokenizer):
        return tokenizer.encode(self.text)

    def prompt_array(self, tokenizer, length):
        ids = self.token_ids(tokenizer)
        if len(ids) < length:
            raise ValueError(
                f"Held-out text has {len(ids)} tokens; need {length}. "
                "Provide a longer corpus."
            )
        return mx.array(ids[:length])

    def filler_text(self, tokenizer, token_budget):
        ids = self.token_ids(tokenizer)
        ids = ids[:token_budget]
        return tokenizer.decode(ids)

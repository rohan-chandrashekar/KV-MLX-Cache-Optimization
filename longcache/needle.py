"""Needle-in-a-haystack retrieval accuracy across context length and insertion depth."""

import random
import re

import mlx.core as mx

QUESTION = (
    "What is the special magic number mentioned in the text? "
    "Answer with only the number."
)


class NeedleHaystack:
    def __init__(self, filler_text, tokenizer, seed=0):
        self.filler_text = filler_text
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)

    def _secret(self):
        return str(self.rng.randint(1000000, 9999999))

    def _filler_for(self, target_tokens):
        ids = self.tokenizer.encode(self.filler_text)
        if len(ids) < target_tokens:
            raise ValueError(
                f"Filler has {len(ids)} tokens; needle test needs {target_tokens}."
            )
        return self.tokenizer.decode(ids[:target_tokens])

    def build(self, target_tokens, depth):
        secret = self._secret()
        needle = f" The special magic number is {secret}. Remember this number. "
        filler = self._filler_for(target_tokens)
        cut = int(len(filler) * depth)
        haystack = filler[:cut] + needle + filler[cut:]
        messages = [{"role": "user", "content": f"{haystack}\n\n{QUESTION}"}]
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return mx.array(ids), secret

    @staticmethod
    def is_correct(output_text, secret):
        found = re.findall(r"\d{4,}", output_text)
        return secret in found


def run_needle_suite(
    runner,
    haystack,
    target_tokens,
    depths,
    answer_tokens,
    kv_bits=None,
    cache_factory=None,
):
    records = []
    correct = 0
    for depth in depths:
        prompt_ids, secret = haystack.build(target_tokens, depth)
        prompt_cache = cache_factory() if cache_factory is not None else None
        result = runner.run_text(
            prompt_ids, answer_tokens, prompt_cache=prompt_cache, kv_bits=kv_bits
        )
        ok = NeedleHaystack.is_correct(result["text"], secret)
        correct += int(ok)
        records.append(
            {
                "depth": depth,
                "secret": secret,
                "answer": result["text"].strip(),
                "correct": ok,
                "prompt_tokens": result["prompt_tokens"],
            }
        )
    return {
        "target_tokens": target_tokens,
        "accuracy": correct / len(depths),
        "records": records,
    }

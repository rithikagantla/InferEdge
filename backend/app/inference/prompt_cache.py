"""Exact-match prompt cache (LRU).

Serving-layer response cache: identical prompts skip the model entirely.
This is deliberately simple — the production analogues are:

  * KV/prefix caching inside the engine (vLLM automatic prefix caching,
    TensorRT-LLM kv-cache reuse) which caches *prefill compute*, and
  * semantic caching (embed the prompt, serve a cached answer above a
    similarity threshold) which also catches paraphrases.

Cache hits are recorded with ~0 latency and $0 marginal cost, which is
what makes the "caching" benchmark mode interesting to compare.
"""
from collections import OrderedDict
from typing import Optional

from .. import config
from .engine import InferenceResult


class PromptCache:
    def __init__(self, max_entries: int = config.PROMPT_CACHE_MAX_ENTRIES) -> None:
        self._store: "OrderedDict[str, InferenceResult]" = OrderedDict()
        self._max = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(prompt: str) -> str:
        return " ".join(prompt.lower().split())

    def get(self, prompt: str) -> Optional[InferenceResult]:
        key = self._key(prompt)
        result = self._store.get(key)
        if result is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return result

    def put(self, prompt: str, result: InferenceResult) -> None:
        key = self._key(prompt)
        self._store[key] = result
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

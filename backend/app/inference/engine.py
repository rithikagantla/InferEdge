"""Inference engine abstraction.

Modes (selected via INFERENCE_MODE):
  mock  — deterministic simulated customer-support model. No GPU or API
          key needed; latency/token behavior mimics a real serving stack.
  nim   — NVIDIA NIM / build.nvidia.com OpenAI-compatible endpoint
          (Nemotron etc.) using NVIDIA_API_KEY.
  local — placeholder for a self-hosted vLLM / TensorRT-LLM / Triton
          backend. See local_engine.py for integration notes.
"""
import abc
from dataclasses import dataclass
from typing import List, Optional

from .. import config


@dataclass
class InferenceResult:
    """Raw output of one generation call, before cost/metrics enrichment."""

    text: str
    model: str
    model_tier: str          # "small" | "large" — used for pricing
    input_tokens: int
    output_tokens: int
    ttft_ms: float           # time to first token (approximated if not streaming)
    latency_ms: float        # total generation latency


class InferenceEngine(abc.ABC):
    """All engines expose single and batched generation.

    Batched generation exists as a first-class method because continuous
    batching is one of the core optimizations being benchmarked: a real
    GPU serving stack (vLLM, TensorRT-LLM, Triton) amortizes weight reads
    and kernel launches across the batch.
    """

    name: str = "base"

    @abc.abstractmethod
    async def generate(self, prompt: str, model_tier: str = "large") -> InferenceResult:
        ...

    async def generate_batch(
        self, prompts: List[str], model_tier: str = "large"
    ) -> List[InferenceResult]:
        """Default batch implementation: concurrent single calls.

        Engines override this when the backend has a true batched path.
        """
        import asyncio

        return list(await asyncio.gather(*(self.generate(p, model_tier) for p in prompts)))


_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """Factory: build the engine for the configured INFERENCE_MODE."""
    global _engine
    if _engine is not None:
        return _engine

    mode = config.INFERENCE_MODE
    if mode == "nim":
        from .nim_engine import NIMEngine

        _engine = NIMEngine()
    elif mode == "local":
        from .local_engine import LocalEngine

        _engine = LocalEngine()
    else:
        from .mock_engine import MockEngine

        _engine = MockEngine()
    return _engine

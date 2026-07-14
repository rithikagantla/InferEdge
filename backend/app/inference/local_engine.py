"""Placeholder for a self-hosted local inference backend.

Where the NVIDIA serving stack plugs in:

  * vLLM            — point an OpenAI-compatible client at
                      http://localhost:8001/v1 (vllm serve <model>).
                      Continuous batching and prefix caching come free.
  * TensorRT-LLM    — compile the model to a TRT engine for the target
                      GPU; serve via Triton's trtllm backend. This is
                      where INT8/FP8 quantization and speculative
                      decoding (draft model or Medusa heads) land.
  * Triton / Dynamo — production serving: dynamic batching config,
                      model ensembles, multi-GPU scheduling, KV-cache
                      aware routing (NVIDIA Dynamo).

Implementing any of these is intentionally out of scope for the MVP;
this class documents the seam so the rest of the platform (metrics,
benchmarks, dashboard) needs zero changes when a real backend arrives.
"""
from .engine import InferenceEngine, InferenceResult


class LocalEngine(InferenceEngine):
    name = "local"

    def __init__(self) -> None:
        raise NotImplementedError(
            "INFERENCE_MODE=local is a placeholder for vLLM / TensorRT-LLM / "
            "Triton. Use INFERENCE_MODE=mock (no deps) or INFERENCE_MODE=nim "
            "(hosted NVIDIA endpoint). See local_engine.py for integration notes."
        )

    async def generate(self, prompt: str, model_tier: str = "large") -> InferenceResult:
        raise NotImplementedError

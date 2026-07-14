"""NVIDIA NIM engine.

Calls an OpenAI-compatible NIM endpoint (hosted build.nvidia.com or a
self-hosted NIM container) with the model from NVIDIA_NIM_MODEL.

Environment:
  NVIDIA_API_KEY      — required
  NVIDIA_NIM_MODEL    — e.g. nvidia/llama-3.1-nemotron-70b-instruct
  NVIDIA_NIM_BASE_URL — defaults to https://integrate.api.nvidia.com/v1

TTFT here is approximated as request latency minus estimated decode time
because we use the non-streaming endpoint; switching to streaming and
timing the first SSE chunk is the drop-in upgrade (see README roadmap).
"""
import time
from typing import List

import httpx

from .. import config
from .engine import InferenceEngine, InferenceResult

SYSTEM_PROMPT = (
    "You are a concise, friendly customer-support assistant. "
    "Resolve the customer's issue in under 120 words."
)


class NIMEngine(InferenceEngine):
    name = "nim"

    def __init__(self) -> None:
        if not config.NVIDIA_API_KEY:
            raise RuntimeError(
                "INFERENCE_MODE=nim requires NVIDIA_API_KEY to be set. "
                "Get a free key at https://build.nvidia.com, or use INFERENCE_MODE=mock."
            )
        self._client = httpx.AsyncClient(
            base_url=config.NVIDIA_NIM_BASE_URL,
            headers={"Authorization": "Bearer " + config.NVIDIA_API_KEY},
            timeout=120.0,
        )

    async def generate(self, prompt: str, model_tier: str = "large") -> InferenceResult:
        # The "small" tier maps to a lighter NIM model for routed traffic;
        # "quantized" falls through to the main model (hosted NIM does not
        # expose a quantization toggle — see local_engine.py notes).
        model = (
            "meta/llama-3.1-8b-instruct"
            if model_tier == "small"
            else config.NVIDIA_NIM_MODEL
        )
        start = time.perf_counter()
        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 256,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", config.estimate_tokens(prompt))
        output_tokens = usage.get("completion_tokens", config.estimate_tokens(text))

        # Non-streaming TTFT approximation: assume decode dominates and
        # scale by share of one token. Replace with real first-chunk
        # timing once streaming is enabled.
        ttft_ms = latency_ms / max(output_tokens, 1)

        return InferenceResult(
            text=text,
            model=model,
            model_tier=model_tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
        )

"""NVIDIA NIM engine.

Calls an OpenAI-compatible NIM endpoint (hosted build.nvidia.com or a
self-hosted NIM container) through the shared streaming client.

Environment:
  NVIDIA_API_KEY          — required
  NVIDIA_NIM_MODEL        — large tier, e.g. nvidia/nemotron-3-ultra-550b-a55b
  NIM_DISABLE_THINKING    — send enable_thinking=false (default true)
  NVIDIA_NIM_SMALL_MODEL  — small tier for routed traffic
  NVIDIA_NIM_BASE_URL     — defaults to https://integrate.api.nvidia.com/v1
  NIM_MAX_CONCURRENCY     — max in-flight requests (free tier is rate limited)
  NIM_MAX_RETRIES         — retries on 429 / 5xx / network errors
"""
from typing import List, Optional

import httpx

from .. import config
from .openai_compat import InferenceAPIError, OpenAICompatEngine

NIMError = InferenceAPIError


class NIMEngine(OpenAICompatEngine):
    name = "nim"
    auth_hint = "check NVIDIA_API_KEY"

    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        if not config.NVIDIA_API_KEY:
            raise RuntimeError(
                "INFERENCE_MODE=nim requires NVIDIA_API_KEY to be set. "
                "Get a free key at https://build.nvidia.com, or use INFERENCE_MODE=mock."
            )
        super().__init__(config.NVIDIA_NIM_BASE_URL, config.NVIDIA_API_KEY, transport)

    def model_for_tier(self, model_tier: str) -> str:
        # "quantized" maps to the large model: hosted NIM has no
        # quantization toggle (see gpu/modal_trtllm.py for real FP8 engines).
        if model_tier == "small":
            return config.NVIDIA_NIM_SMALL_MODEL
        return config.NVIDIA_NIM_MODEL

    def _payload(self, prompt: str, model: str, history: Optional[List[dict]] = None) -> dict:
        payload = super()._payload(prompt, model, history)
        if config.NIM_DISABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    @property
    def max_retries(self) -> int:
        return config.NIM_MAX_RETRIES

    @property
    def max_concurrency(self) -> int:
        return config.NIM_MAX_CONCURRENCY

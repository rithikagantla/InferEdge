"""Self-hosted engine: any OpenAI-compatible server you run yourself.

Works with `vllm serve` and `trtllm-serve` (TensorRT-LLM), which both
expose /v1/chat/completions. gpu/modal_vllm.py and gpu/modal_trtllm.py
start these servers on a cloud GPU; `modal serve gpu/modal_vllm.py`
prints a URL to put in LOCAL_BASE_URL.

Environment:
  LOCAL_BASE_URL        — e.g. http://localhost:8001/v1
  LOCAL_MODEL           — model name the server was started with
  LOCAL_API_KEY         — optional, if the server was started with one
  LOCAL_SMALL_BASE_URL  — optional second server for routed small-tier
  LOCAL_SMALL_MODEL       traffic; without it both tiers hit LOCAL_MODEL,
                          so routing changes nothing in local mode
"""
from typing import List, Optional

import httpx

from .. import config
from .openai_compat import OpenAICompatEngine


class LocalEngine(OpenAICompatEngine):
    name = "local"
    auth_hint = "check LOCAL_API_KEY"

    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        if not config.LOCAL_BASE_URL:
            raise RuntimeError(
                "INFERENCE_MODE=local requires LOCAL_BASE_URL (an OpenAI-compatible "
                "server such as `vllm serve` or `trtllm-serve`). See docs/SELF_HOSTED.md."
            )
        super().__init__(config.LOCAL_BASE_URL, config.LOCAL_API_KEY, transport)
        self._small: Optional[OpenAICompatEngine] = None
        if config.LOCAL_SMALL_BASE_URL:
            self._small = _SingleModel(
                config.LOCAL_SMALL_BASE_URL, config.LOCAL_SMALL_MODEL, config.LOCAL_API_KEY
            )

    def model_for_tier(self, model_tier: str) -> str:
        return config.LOCAL_MODEL

    @property
    def max_concurrency(self) -> int:
        return config.LOCAL_MAX_CONCURRENCY

    async def generate(self, prompt: str, model_tier: str = "large",
                       history: Optional[List[dict]] = None):
        if model_tier == "small" and self._small is not None:
            return await self._small.generate(prompt, "small", history)
        return await super().generate(prompt, model_tier, history)


class _SingleModel(OpenAICompatEngine):
    name = "local-small"

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        super().__init__(base_url, api_key)
        self._model = model

    def model_for_tier(self, model_tier: str) -> str:
        return self._model

    @property
    def max_concurrency(self) -> int:
        return config.LOCAL_MAX_CONCURRENCY

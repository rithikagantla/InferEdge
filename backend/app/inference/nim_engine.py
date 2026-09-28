"""NVIDIA NIM engine.

Calls an OpenAI-compatible NIM endpoint (hosted build.nvidia.com or a
self-hosted NIM container).

Environment:
  NVIDIA_API_KEY          — required
  NVIDIA_NIM_MODEL        — large tier, e.g. nvidia/llama-3.1-nemotron-70b-instruct
  NVIDIA_NIM_SMALL_MODEL  — small tier for routed traffic
  NVIDIA_NIM_BASE_URL     — defaults to https://integrate.api.nvidia.com/v1
  NIM_MAX_CONCURRENCY     — max in-flight requests (free tier is rate limited)
  NIM_MAX_RETRIES         — retries on 429 / 5xx / network errors

Requests are streamed (SSE) so time-to-first-token is measured directly:
the clock stops at the first chunk that carries content, not the first
byte (NIM sends a role-only chunk before any tokens).
"""
import asyncio
import json
import logging
import random
import time
from typing import Optional

import httpx

from .. import config
from .engine import InferenceEngine, InferenceResult

logger = logging.getLogger("inferedge.nim")

SYSTEM_PROMPT = (
    "You are a concise, friendly customer-support assistant. "
    "Resolve the customer's issue in under 120 words."
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_BACKOFF_BASE_S = 1.0
_BACKOFF_MAX_S = 30.0


class NIMError(RuntimeError):
    pass


class _Retryable(Exception):
    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class NIMEngine(InferenceEngine):
    name = "nim"

    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        if not config.NVIDIA_API_KEY:
            raise RuntimeError(
                "INFERENCE_MODE=nim requires NVIDIA_API_KEY to be set. "
                "Get a free key at https://build.nvidia.com, or use INFERENCE_MODE=mock."
            )
        self._client = httpx.AsyncClient(
            base_url=config.NVIDIA_NIM_BASE_URL,
            headers={"Authorization": "Bearer " + config.NVIDIA_API_KEY},
            timeout=httpx.Timeout(120.0, connect=10.0),
            transport=transport,
        )
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None
        # Not every NIM model accepts stream_options; drop it after the
        # first rejection and fall back to estimated token counts.
        self._include_usage = True

    def _limiter(self) -> asyncio.Semaphore:
        # On Python 3.9 a Semaphore binds to the loop current at creation,
        # so build it inside the running loop.
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(max(1, config.NIM_MAX_CONCURRENCY))
            self._semaphore_loop = loop
        return self._semaphore

    @staticmethod
    def model_for_tier(model_tier: str) -> str:
        # "quantized" maps to the large model: hosted NIM has no
        # quantization toggle (see local_engine.py for where TensorRT-LLM
        # INT8/FP8 engines would plug in).
        if model_tier == "small":
            return config.NVIDIA_NIM_SMALL_MODEL
        return config.NVIDIA_NIM_MODEL

    async def generate(self, prompt: str, model_tier: str = "large") -> InferenceResult:
        model = self.model_for_tier(model_tier)
        async with self._limiter():
            for attempt in range(config.NIM_MAX_RETRIES + 1):
                try:
                    return await self._stream_once(prompt, model, model_tier)
                except _Retryable as exc:
                    if attempt == config.NIM_MAX_RETRIES:
                        raise NIMError(
                            "%s failed after %d retries: %s"
                            % (model, config.NIM_MAX_RETRIES, exc)
                        )
                    delay = exc.retry_after
                    if delay is None:
                        delay = min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * 2 ** attempt)
                        delay *= random.uniform(0.8, 1.2)
                    logger.warning(
                        "NIM retry %d/%d for %s in %.1fs: %s",
                        attempt + 1, config.NIM_MAX_RETRIES, model, delay, exc,
                    )
                    await asyncio.sleep(delay)
        raise NIMError("unreachable")  # pragma: no cover

    async def _stream_once(self, prompt: str, model: str, model_tier: str) -> InferenceResult:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 256,
            "temperature": 0.3,
            "stream": True,
        }
        if self._include_usage:
            payload["stream_options"] = {"include_usage": True}

        start = time.perf_counter()
        ttft_ms: Optional[float] = None
        parts = []
        usage: dict = {}
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    self._raise_for_status(resp, body, model)
                    # _raise_for_status returned: stream_options was rejected.
                    self._include_usage = False
                    raise _Retryable("stream_options unsupported; retrying without it", 0.0)

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start) * 1000
                            parts.append(content)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise _Retryable("%s: %s" % (type(exc).__name__, exc))

        latency_ms = (time.perf_counter() - start) * 1000
        text = "".join(parts).strip()
        if not text:
            raise _Retryable("empty completion")

        return InferenceResult(
            text=text,
            model=model,
            model_tier=model_tier,
            input_tokens=usage.get("prompt_tokens") or config.estimate_tokens(SYSTEM_PROMPT + prompt),
            output_tokens=usage.get("completion_tokens") or config.estimate_tokens(text),
            ttft_ms=ttft_ms if ttft_ms is not None else latency_ms,
            latency_ms=latency_ms,
        )

    def _raise_for_status(self, resp: httpx.Response, body: str, model: str) -> None:
        """Raise for a non-200 response, or return if the only problem was
        an unsupported stream_options field (caller retries without it)."""
        status = resp.status_code
        if status in _RETRYABLE_STATUS:
            retry_after = resp.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else None
            except ValueError:
                delay = None
            raise _Retryable("HTTP %d: %s" % (status, body), delay)
        if status in (400, 422) and self._include_usage and "stream_options" in body:
            return
        if status in (401, 403):
            raise NIMError("HTTP %d from NIM — check NVIDIA_API_KEY. %s" % (status, body))
        if status == 404:
            raise NIMError(
                "HTTP 404 for model %r — it may have been removed from the hosted "
                "catalog; set NVIDIA_NIM_MODEL / NVIDIA_NIM_SMALL_MODEL. %s" % (model, body)
            )
        raise NIMError("HTTP %d from NIM: %s" % (status, body))

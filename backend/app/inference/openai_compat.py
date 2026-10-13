"""Streaming client for OpenAI-compatible chat-completion servers.

Hosted NVIDIA NIM, `vllm serve` and `trtllm-serve` all expose the same
/v1/chat/completions API, so one client serves every real engine and the
serving sweep (scripts/sweep_serving.py). Measuring all of them through
the same code path is what makes their numbers comparable.

Requests are streamed (SSE) so time-to-first-token is measured directly:
the clock stops at the first chunk that carries content, not the first
byte (servers often send a role-only chunk before any tokens).
"""
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

from .. import config
from .engine import InferenceEngine, InferenceResult

logger = logging.getLogger("inferedge.openai_compat")

SYSTEM_PROMPT = (
    "You are a concise, friendly customer-support assistant. "
    "Resolve the customer's issue in under 120 words."
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_BACKOFF_BASE_S = 1.0
_BACKOFF_MAX_S = 30.0


class InferenceAPIError(RuntimeError):
    """Non-retryable failure from the server (bad key, unknown model, ...)."""


class RetryableError(Exception):
    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class StreamOptionsUnsupported(Exception):
    """The server rejected `stream_options`; resend without it."""


@dataclass
class StreamResult:
    text: str
    ttft_ms: Optional[float]
    latency_ms: float
    # Arrival time (ms since request start) of every chunk that carried
    # content; gaps between them are the inter-token latencies.
    chunk_times_ms: List[float] = field(default_factory=list)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


async def stream_chat_completion(
    client: httpx.AsyncClient,
    payload: dict,
    auth_hint: str = "check the API key",
) -> StreamResult:
    """One streamed request, no retries. Raises RetryableError,
    StreamOptionsUnsupported or InferenceAPIError on failure."""
    payload = dict(payload, stream=True)
    start = time.perf_counter()
    ttft_ms: Optional[float] = None
    parts: List[str] = []
    chunk_times: List[float] = []
    usage: dict = {}
    try:
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:500]
                _raise_for_status(resp, body, payload, auth_hint)

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
                        now_ms = (time.perf_counter() - start) * 1000
                        if ttft_ms is None:
                            ttft_ms = now_ms
                        chunk_times.append(now_ms)
                        parts.append(content)
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        raise RetryableError("%s: %s" % (type(exc).__name__, exc))

    return StreamResult(
        text="".join(parts).strip(),
        ttft_ms=ttft_ms,
        latency_ms=(time.perf_counter() - start) * 1000,
        chunk_times_ms=chunk_times,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def _raise_for_status(resp: httpx.Response, body: str, payload: dict, auth_hint: str) -> None:
    status = resp.status_code
    model = payload.get("model")
    if status in _RETRYABLE_STATUS:
        retry_after = resp.headers.get("retry-after")
        try:
            delay = float(retry_after) if retry_after else None
        except ValueError:
            delay = None
        raise RetryableError("HTTP %d: %s" % (status, body), delay)
    if status in (400, 422) and "stream_options" in payload and "stream_options" in body:
        raise StreamOptionsUnsupported(body)
    if status in (401, 403):
        raise InferenceAPIError("HTTP %d — %s. %s" % (status, auth_hint, body))
    if status == 404:
        raise InferenceAPIError(
            "HTTP 404 for model %r — it may have been removed from the hosted "
            "catalog or not be loaded on this server. %s" % (model, body)
        )
    raise InferenceAPIError("HTTP %d: %s" % (status, body))


class OpenAICompatEngine(InferenceEngine):
    """Engine for any OpenAI-compatible server. Subclasses choose the base
    URL, credentials, and which model serves each tier."""

    name = "openai-compat"
    auth_hint = "check the API key"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=10.0),
            transport=transport,
        )
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None
        # Not every server accepts stream_options; drop it after the first
        # rejection and fall back to estimated token counts.
        self._include_usage = True

    def model_for_tier(self, model_tier: str) -> str:
        raise NotImplementedError

    @property
    def max_retries(self) -> int:
        return 5

    @property
    def max_concurrency(self) -> int:
        return 4

    def _limiter(self) -> asyncio.Semaphore:
        # On Python 3.9 a Semaphore binds to the loop current at creation,
        # so build it inside the running loop.
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(max(1, self.max_concurrency))
            self._semaphore_loop = loop
        return self._semaphore

    def _payload(self, prompt: str, model: str) -> dict:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 256,
            "temperature": 0.3,
        }
        if self._include_usage:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def generate(self, prompt: str, model_tier: str = "large") -> InferenceResult:
        model = self.model_for_tier(model_tier)
        async with self._limiter():
            attempt = 0
            while True:
                try:
                    result = await stream_chat_completion(
                        self._client, self._payload(prompt, model), self.auth_hint
                    )
                    if not result.text:
                        raise RetryableError("empty completion")
                    break
                except StreamOptionsUnsupported:
                    self._include_usage = False
                    continue
                except RetryableError as exc:
                    if attempt >= self.max_retries:
                        raise InferenceAPIError(
                            "%s failed after %d retries: %s" % (model, self.max_retries, exc)
                        )
                    delay = exc.retry_after
                    if delay is None:
                        delay = min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * 2 ** attempt)
                        delay *= random.uniform(0.8, 1.2)
                    attempt += 1
                    logger.warning(
                        "%s retry %d/%d for %s in %.1fs: %s",
                        self.name, attempt, self.max_retries, model, delay, exc,
                    )
                    await asyncio.sleep(delay)

        return InferenceResult(
            text=result.text,
            model=model,
            model_tier=model_tier,
            input_tokens=result.prompt_tokens or config.estimate_tokens(SYSTEM_PROMPT + prompt),
            output_tokens=result.completion_tokens or config.estimate_tokens(result.text),
            ttft_ms=result.ttft_ms if result.ttft_ms is not None else result.latency_ms,
            latency_ms=result.latency_ms,
        )

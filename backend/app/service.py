"""Service layer: applies the selected optimization strategy to a
request, runs inference, and records metrics.

Optimization modes
------------------
baseline  — one prompt, one large-model call. The control group.
caching   — exact-match LRU response cache in front of the engine.
routing   — semantic router picks a model tier (small vs large) per
            intent, cutting cost/latency for easy traffic.
quantized — simulated INT8/FP8 serving profile of the large model.
            Placeholder for a real TensorRT-LLM quantized engine.
batching  — exposed via batch_chat(); requests share the GPU decode
            loop (continuous batching) so throughput scales.
"""
import time
from typing import List, Optional

from . import config
from .inference import get_engine
from .inference.engine import InferenceResult
from .inference.prompt_cache import PromptCache
from .metrics import tracker
from .routing import get_router
from .schemas import BatchChatResponse, ChatResponse, RequestMetrics

prompt_cache = PromptCache()


def _build_metrics(
    result: InferenceResult,
    mode: str,
    cached: bool = False,
    route: Optional[dict] = None,
    history_turns: int = 0,
) -> RequestMetrics:
    route = route or {}
    tier = config.MODEL_TIERS.get(result.model_tier)
    cost = 0.0 if cached else config.estimate_cost_usd(
        result.model_tier, result.input_tokens, result.output_tokens
    )
    tokens_per_sec = (
        0.0 if cached or result.latency_ms <= 0
        else result.output_tokens / (result.latency_ms / 1000)
    )
    return RequestMetrics(
        mode=mode,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        ttft_ms=round(result.ttft_ms, 3),
        latency_ms=round(result.latency_ms, 3),
        tokens_per_sec=round(tokens_per_sec, 2),
        cost_usd=round(cost, 8),
        cached=cached,
        category=route.get("category"),
        router_backend=route.get("backend"),
        router_time_ms=route.get("route_time_ms"),
        tier=result.model_tier,
        input_price_per_1m=tier.input_cost_per_1m if tier else None,
        output_price_per_1m=tier.output_cost_per_1m if tier else None,
        route_confidence=route.get("confidence"),
        route_scores=route.get("scores"),
        history_turns=history_turns,
    )


async def chat(prompt: str, mode: str = "baseline", record: bool = True,
               history: Optional[List[dict]] = None) -> ChatResponse:
    engine = get_engine()
    history = history or []
    turns = len(history)

    # The response cache is keyed on the prompt alone, so it only serves
    # single-turn requests: "what about the refund?" means different things
    # in different conversations.
    if mode == "caching" and not history:
        hit = prompt_cache.get(prompt)
        if hit is not None:
            start = time.perf_counter()
            lookup_ms = (time.perf_counter() - start) * 1000 + 0.05
            cached_result = InferenceResult(
                text=hit.text,
                model=hit.model + " (cache hit)",
                model_tier=hit.model_tier,
                input_tokens=hit.input_tokens,
                output_tokens=hit.output_tokens,
                ttft_ms=lookup_ms,
                latency_ms=lookup_ms,
            )
            metrics = _build_metrics(cached_result, mode, cached=True)
            if record:
                tracker.record(metrics)
            return ChatResponse(response=hit.text, metrics=metrics)
        result = await engine.generate(prompt, "large")
        prompt_cache.put(prompt, result)
        metrics = _build_metrics(result, mode)

    elif mode == "routing":
        # Route on the newest message: that's what the next answer is about.
        route = get_router().classify(prompt)
        result = await engine.generate(prompt, route["model_tier"], history)
        metrics = _build_metrics(result, mode, route=route, history_turns=turns)

    elif mode == "quantized":
        # Mock-only: a faster serving profile simulating an INT8/FP8
        # TensorRT-LLM engine. The API rejects this mode outside mock
        # because hosted NIM exposes no quantization toggle.
        result = await engine.generate(prompt, "quantized", history)
        metrics = _build_metrics(result, mode, history_turns=turns)

    else:  # baseline, or caching with history (cache bypassed)
        result = await engine.generate(prompt, "large", history)
        metrics = _build_metrics(result, mode, history_turns=turns)

    if record:
        tracker.record(metrics)
    return ChatResponse(response=result.text, metrics=metrics)


async def batch_chat(prompts: List[str], record: bool = True) -> BatchChatResponse:
    engine = get_engine()
    start = time.perf_counter()
    results = await engine.generate_batch(prompts, "large")
    wall_ms = (time.perf_counter() - start) * 1000

    responses = []
    for result in results:
        metrics = _build_metrics(result, "batching")
        if record:
            tracker.record(metrics)
        responses.append(ChatResponse(response=result.text, metrics=metrics))

    return BatchChatResponse(
        responses=responses,
        batch_size=len(prompts),
        wall_time_ms=round(wall_ms, 2),
        requests_per_sec=round(len(prompts) / (wall_ms / 1000), 2) if wall_ms > 0 else 0.0,
    )

"""Benchmark runner: measures each optimization mode over N prompts.

Traffic shape per mode:
  baseline / routing / quantized — sequential requests (a single agent
      conversation loop; latency is what matters).
  batching — prompts grouped into batches of BATCH_SIZE and served
      through the engine's batched path (throughput is what matters).
  caching — prompts drawn from a small "hot FAQ" pool with repetition,
      which is what real support traffic looks like and what makes a
      response cache pay off.
"""
import asyncio
import random
import time
from datetime import datetime, timezone
from itertools import islice, cycle
from typing import List, Optional

from . import config, db, service
from .metrics import percentile
from .sample_prompts import SAMPLE_PROMPTS
from .schemas import OPTIMIZATION_MODES, BenchmarkResult, ChatResponse, ModeResult

BATCH_SIZE = 8
HOT_FAQ_POOL = 12  # caching mode samples from this many distinct prompts

MODE_NOTES = {
    "baseline": "Sequential single requests to the large model (control).",
    "batching": "Batches of %d served concurrently — simulates GPU continuous batching." % BATCH_SIZE,
    "caching": "Zipf-like traffic from a hot pool of %d FAQs through an LRU response cache." % HOT_FAQ_POOL,
    "routing": "CUDA/CPU semantic router sends easy intents to a small model tier.",
    "quantized": "Simulated INT8/FP8 profile — placeholder for a TensorRT-LLM quantized engine.",
}

# Against a hosted endpoint the server's batching is out of our hands, so
# "batching" measures client-side concurrency (capped by NIM_MAX_CONCURRENCY).
NIM_MODE_NOTES = {
    "batching": "Batches of %d sent concurrently to hosted NIM (client-side concurrency; "
    "server-side batching is NIM's)." % BATCH_SIZE,
}

# Modes that only mean something in the simulator: hosted NIM exposes no
# quantization toggle, so a "quantized" row would just be baseline relabeled.
MOCK_ONLY_MODES = {"quantized"}


def _prompts_for(mode: str, n: int) -> List[str]:
    if mode == "caching":
        rng = random.Random(1234)
        return rng.choices(SAMPLE_PROMPTS[:HOT_FAQ_POOL], k=n)
    return list(islice(cycle(SAMPLE_PROMPTS), n))


def _aggregate(mode: str, responses: List[ChatResponse], wall_time_s: float) -> ModeResult:
    rows = [r.metrics for r in responses]
    latencies = [m.latency_ms for m in rows]
    non_cached = [m for m in rows if not m.cached]
    input_tokens = sum(m.input_tokens for m in rows)
    output_tokens = sum(m.output_tokens for m in rows)
    cost = sum(m.cost_usd for m in rows)
    total_tokens = input_tokens + output_tokens
    return ModeResult(
        mode=mode,
        num_requests=len(rows),
        wall_time_s=round(wall_time_s, 3),
        requests_per_sec=round(len(rows) / wall_time_s, 2) if wall_time_s > 0 else 0.0,
        p50_latency_ms=round(percentile(latencies, 50), 2),
        p95_latency_ms=round(percentile(latencies, 95), 2),
        avg_ttft_ms=round(sum(m.ttft_ms for m in rows) / len(rows), 2),
        avg_tokens_per_sec=round(
            sum(m.tokens_per_sec for m in non_cached) / len(non_cached), 2
        ) if non_cached else 0.0,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_cost_usd=round(cost, 6),
        est_cost_per_1m_tokens=round(cost / total_tokens * 1_000_000, 4) if total_tokens else 0.0,
        cache_hit_rate=round(
            sum(1 for m in rows if m.cached) / len(rows), 3
        ) if mode == "caching" else None,
        notes=(
            NIM_MODE_NOTES.get(mode, MODE_NOTES.get(mode))
            if config.INFERENCE_MODE == "nim"
            else MODE_NOTES.get(mode)
        ),
    )


async def _run_mode(mode: str, num_prompts: int) -> ModeResult:
    prompts = _prompts_for(mode, num_prompts)
    responses: List[ChatResponse] = []
    start = time.perf_counter()

    if mode == "batching":
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = await service.batch_chat(prompts[i : i + BATCH_SIZE])
            responses.extend(batch.responses)
    else:
        if mode == "caching":
            service.prompt_cache.clear()  # fair start for every run
        for prompt in prompts:
            responses.append(await service.chat(prompt, mode=mode))

    wall_time_s = time.perf_counter() - start
    return _aggregate(mode, responses, wall_time_s)


async def run_benchmark(
    num_prompts: int = 20, modes: Optional[List[str]] = None
) -> BenchmarkResult:
    selected = [m for m in (modes or OPTIMIZATION_MODES) if m in OPTIMIZATION_MODES]
    if config.INFERENCE_MODE != "mock":
        selected = [m for m in selected if m not in MOCK_ONLY_MODES]
    if not selected:
        raise ValueError("no valid modes; choose from %s" % OPTIMIZATION_MODES)

    results = []
    for mode in selected:
        results.append(await _run_mode(mode, num_prompts))
        await asyncio.sleep(0)  # yield to the event loop between modes

    benchmark = BenchmarkResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        inference_mode=config.INFERENCE_MODE,
        num_prompts=num_prompts,
        results=results,
    )
    benchmark.id = db.save_benchmark(benchmark)
    return benchmark

#!/usr/bin/env python3
"""Serving sweep: throughput vs latency across concurrency levels.

Drives any OpenAI-compatible server (vllm serve, trtllm-serve, hosted
NIM) with a closed-loop load: at concurrency C, C workers each send
requests back to back, so the server always has C requests in flight and
its continuous batcher runs at batch size ~C. For each level it records:

  TTFT   time to first token (prefill + queueing)
  TPOT   time per output token after the first = (e2e - TTFT) / (tokens - 1)
  ITL    inter-token latency: gaps between consecutive streamed chunks
  output tokens/s across all requests, requests/s, end-to-end latency

Usage (server already running):
    python scripts/sweep_serving.py --base-url http://localhost:8000/v1 \
        --model NousResearch/Meta-Llama-3-8B-Instruct --label vllm-fp16-L4 --ignore-eos

gpu/modal_vllm.py and gpu/modal_trtllm.py run this inside the GPU
container against localhost, so network round trips stay out of the
numbers.
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from itertools import cycle
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import httpx  # noqa: E402

from app.inference.openai_compat import (  # noqa: E402
    SYSTEM_PROMPT,
    InferenceAPIError,
    RetryableError,
    StreamOptionsUnsupported,
    StreamResult,
    stream_chat_completion,
)
from app.sample_prompts import SAMPLE_PROMPTS  # noqa: E402

DEFAULT_LEVELS = [1, 2, 4, 8, 16, 32, 64]

CSV_COLUMNS = [
    "concurrency", "requests", "errors", "wall_s", "requests_per_sec",
    "output_tokens_per_sec", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms",
    "tpot_p95_ms", "itl_p50_ms", "itl_p99_ms", "e2e_p50_ms", "e2e_p95_ms",
    "avg_output_tokens",
]


def percentile(values: List[float], pct: float) -> Optional[float]:
    """Linear-interpolated percentile (same definition as numpy's default)."""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def request_stats(result: StreamResult) -> dict:
    """Per-request metrics. Output tokens come from the server's usage
    report when available, otherwise one token per streamed chunk."""
    tokens = result.completion_tokens or len(result.chunk_times_ms)
    ttft = result.ttft_ms if result.ttft_ms is not None else result.latency_ms
    tpot = (result.latency_ms - ttft) / (tokens - 1) if tokens > 1 else None
    times = result.chunk_times_ms
    itls = [b - a for a, b in zip(times, times[1:])]
    return {
        "ttft_ms": ttft,
        "e2e_ms": result.latency_ms,
        "output_tokens": tokens,
        "tpot_ms": tpot,
        "itls_ms": itls,
    }


def summarize_level(concurrency: int, rows: List[dict], errors: int, wall_s: float) -> dict:
    def r(v, nd=2):
        return None if v is None else round(v, nd)

    ttfts = [x["ttft_ms"] for x in rows]
    tpots = [x["tpot_ms"] for x in rows if x["tpot_ms"] is not None]
    itls = [g for x in rows for g in x["itls_ms"]]
    e2es = [x["e2e_ms"] for x in rows]
    total_out = sum(x["output_tokens"] for x in rows)
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "errors": errors,
        "wall_s": r(wall_s, 3),
        "requests_per_sec": r(len(rows) / wall_s if wall_s > 0 else 0.0, 3),
        "output_tokens_per_sec": r(total_out / wall_s if wall_s > 0 else 0.0, 1),
        "ttft_p50_ms": r(percentile(ttfts, 50)),
        "ttft_p95_ms": r(percentile(ttfts, 95)),
        "tpot_p50_ms": r(percentile(tpots, 50)),
        "tpot_p95_ms": r(percentile(tpots, 95)),
        "itl_p50_ms": r(percentile(itls, 50)),
        "itl_p99_ms": r(percentile(itls, 99)),
        "e2e_p50_ms": r(percentile(e2es, 50)),
        "e2e_p95_ms": r(percentile(e2es, 95)),
        "avg_output_tokens": r(statistics.mean([x["output_tokens"] for x in rows]) if rows else None, 1),
    }


class _Client:
    """Builds payloads and remembers whether the server takes stream_options."""

    def __init__(self, http: httpx.AsyncClient, model: str, max_tokens: int, ignore_eos: bool,
                 extra_body: Optional[dict] = None):
        self.http = http
        self.extra_body = extra_body or {}
        self.model = model
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.include_usage = True

    async def one(self, prompt: str) -> StreamResult:
        while True:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
            }
            if self.ignore_eos:
                # vLLM / TensorRT-LLM extension: always generate max_tokens,
                # so every request does the same amount of decode work.
                payload["ignore_eos"] = True
            if self.include_usage:
                payload["stream_options"] = {"include_usage": True}
            payload.update(self.extra_body)
            try:
                return await stream_chat_completion(self.http, payload)
            except StreamOptionsUnsupported:
                self.include_usage = False


async def run_level(client: _Client, concurrency: int, num_requests: int) -> dict:
    prompts = cycle(SAMPLE_PROMPTS)
    remaining = [num_requests]
    rows: List[dict] = []
    errors: List[str] = []

    async def worker() -> None:
        while remaining[0] > 0:
            remaining[0] -= 1
            prompt = next(prompts)
            try:
                result = await client.one(prompt)
                if not result.chunk_times_ms:
                    raise RetryableError("empty completion")
                rows.append(request_stats(result))
            except (RetryableError, InferenceAPIError, ValueError) as exc:
                errors.append(str(exc)[:300])

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall_s = time.perf_counter() - start
    summary = summarize_level(concurrency, rows, len(errors), wall_s)
    if errors:
        summary["first_error"] = errors[0]
    return summary


def requests_for_level(concurrency: int, min_requests: int, per_worker: int) -> int:
    return max(min_requests, concurrency * per_worker)


async def run_sweep(
    base_url: str,
    model: str,
    label: str,
    levels: List[int] = DEFAULT_LEVELS,
    max_tokens: int = 256,
    ignore_eos: bool = False,
    min_requests: int = 8,
    per_worker: int = 3,
    warmup: int = 3,
    api_key: str = "",
    log=print,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    extra_body: Optional[dict] = None,
) -> dict:
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    limits = httpx.Limits(max_connections=max(levels) + 8, max_keepalive_connections=max(levels) + 8)
    async with httpx.AsyncClient(
        base_url=base_url, headers=headers, limits=limits,
        timeout=httpx.Timeout(600.0, connect=30.0), transport=transport,
    ) as http:
        client = _Client(http, model, max_tokens, ignore_eos, extra_body)
        log("warm-up: %d requests" % warmup)
        for prompt in SAMPLE_PROMPTS[:warmup]:
            await client.one(prompt)

        results = []
        for c in levels:
            n = requests_for_level(c, min_requests, per_worker)
            level = await run_level(client, c, n)
            results.append(level)
            log(
                "c=%-3d  %4.1f req/s  %7.1f out tok/s  TTFT p50 %7.1f ms  TPOT p50 %6.2f ms  errors %d"
                % (c, level["requests_per_sec"], level["output_tokens_per_sec"],
                   level["ttft_p50_ms"] or 0, level["tpot_p50_ms"] or 0, level["errors"])
            )

    return {
        "label": label,
        "model": model,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "extra_body": extra_body or {},
        "stream_usage_reported": client.include_usage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "levels": results,
    }


def markdown_table(sweep: dict) -> str:
    lines = [
        "| Concurrency | Output tok/s | Req/s | TTFT p50 (ms) | TTFT p95 (ms) "
        "| TPOT p50 (ms) | ITL p99 (ms) | E2E p50 (ms) | Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lv in sweep["levels"]:
        lines.append(
            "| {concurrency} | {output_tokens_per_sec:.0f} | {requests_per_sec:.2f} "
            "| {ttft_p50_ms:.0f} | {ttft_p95_ms:.0f} | {tpot_p50_ms:.1f} "
            "| {itl_p99_ms:.1f} | {e2e_p50_ms:.0f} | {errors} |".format(
                **{k: (v if v is not None else float("nan")) for k, v in lv.items()}
            )
        )
    return "\n".join(lines) + "\n"


def write_outputs(sweep: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "sweep_%s" % sweep["label"])
    with open(base + ".json", "w") as f:
        json.dump(sweep, f, indent=2)
    with open(base + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep["levels"])
    with open(base + ".md", "w") as f:
        f.write(markdown_table(sweep))
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Throughput-vs-latency serving sweep")
    parser.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, help="e.g. vllm-fp16-L4")
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ignore-eos", action="store_true",
                        help="force max-tokens output (vLLM / TensorRT-LLM only)")
    parser.add_argument("--min-requests", type=int, default=8)
    parser.add_argument("--per-worker", type=int, default=3,
                        help="requests per worker at each level")
    parser.add_argument("--api-key", default=os.getenv("SWEEP_API_KEY", ""))
    parser.add_argument("--extra-body", type=json.loads, default=None,
                        help='JSON merged into every request, e.g. '
                             '\'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    parser.add_argument("--out-dir", default="benchmark_results/gpu")
    args = parser.parse_args()

    sweep = asyncio.run(run_sweep(
        args.base_url, args.model, args.label, args.levels, args.max_tokens,
        args.ignore_eos, args.min_requests, args.per_worker, api_key=args.api_key,
        extra_body=args.extra_body,
    ))
    base = write_outputs(sweep, args.out_dir)
    print("\n" + markdown_table(sweep))
    print("Saved: %s.json / .csv / .md" % base)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""In-memory metrics tracker with percentile aggregation.

Every request that goes through the service layer records a
RequestMetrics row here; GET /metrics aggregates the most recent
METRICS_WINDOW rows (p50/p95 latency, tokens/sec, req/s, cost per 1M
tokens) overall and per optimization mode.
"""
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from . import config
from .schemas import RequestMetrics

logger = logging.getLogger("inferedge.metrics")


def percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile; avoids a numpy dependency here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[k]


class MetricsTracker:
    def __init__(self, window: int = config.METRICS_WINDOW) -> None:
        self._lock = threading.Lock()
        # (unix_timestamp, metrics)
        self._records: Deque[Tuple[float, RequestMetrics]] = deque(maxlen=window)
        self._total_requests = 0

    def record(self, metrics: RequestMetrics) -> None:
        with self._lock:
            self._records.append((time.time(), metrics))
            self._total_requests += 1
        logger.info(
            "mode=%s model=%s latency=%.1fms ttft=%.1fms tok/s=%.1f in=%d out=%d cost=$%.6f cached=%s",
            metrics.mode, metrics.model, metrics.latency_ms, metrics.ttft_ms,
            metrics.tokens_per_sec, metrics.input_tokens, metrics.output_tokens,
            metrics.cost_usd, metrics.cached,
        )

    @staticmethod
    def _aggregate(rows: List[RequestMetrics], window_seconds: Optional[float]) -> dict:
        latencies = [r.latency_ms for r in rows]
        ttfts = [r.ttft_ms for r in rows]
        # Cache hits report 0 tok/s (no generation happened) — exclude
        # them so the average reflects actual decode speed.
        generated = [r for r in rows if not r.cached]
        input_tokens = sum(r.input_tokens for r in rows)
        output_tokens = sum(r.output_tokens for r in rows)
        cost = sum(r.cost_usd for r in rows)
        total_tokens = input_tokens + output_tokens
        rps = len(rows) / window_seconds if window_seconds and window_seconds > 0 else None
        return {
            "requests": len(rows),
            "p50_latency_ms": round(percentile(latencies, 50), 2),
            "p95_latency_ms": round(percentile(latencies, 95), 2),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else 0.0,
            "avg_tokens_per_sec": round(
                sum(r.tokens_per_sec for r in generated) / len(generated), 2
            ) if generated else 0.0,
            "requests_per_sec": round(rps, 3) if rps is not None else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost_usd": round(cost, 6),
            "est_cost_per_1m_tokens": round(cost / total_tokens * 1_000_000, 4)
            if total_tokens
            else 0.0,
            "cache_hit_rate": round(
                sum(1 for r in rows if r.cached) / len(rows), 3
            ) if rows else 0.0,
        }

    def snapshot(self) -> dict:
        with self._lock:
            items = list(self._records)
            total = self._total_requests
        rows = [m for _, m in items]
        window_seconds = (items[-1][0] - items[0][0]) if len(items) > 1 else None

        by_mode: Dict[str, List[RequestMetrics]] = {}
        for r in rows:
            by_mode.setdefault(r.mode, []).append(r)

        router_times = [r.router_time_ms for r in rows if r.router_time_ms is not None]
        return {
            "inference_mode": config.INFERENCE_MODE,
            "total_requests": total,
            "window_size": len(rows),
            "overall": self._aggregate(rows, window_seconds),
            "by_mode": {mode: self._aggregate(mrows, None) for mode, mrows in by_mode.items()},
            "avg_router_time_ms": round(sum(router_times) / len(router_times), 4)
            if router_times
            else None,
            "recent": [r.model_dump() for r in rows[-10:]][::-1],
        }


tracker = MetricsTracker()

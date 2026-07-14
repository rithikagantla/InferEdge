"""Pydantic request/response schemas for the InferEdge API."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

OPTIMIZATION_MODES = ["baseline", "batching", "caching", "routing", "quantized"]


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    mode: str = Field("baseline", description="baseline | caching | routing | quantized")


class RequestMetrics(BaseModel):
    mode: str
    model: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    latency_ms: float
    tokens_per_sec: float
    cost_usd: float
    cached: bool = False
    category: Optional[str] = None
    router_backend: Optional[str] = None
    router_time_ms: Optional[float] = None


class ChatResponse(BaseModel):
    response: str
    metrics: RequestMetrics


class BatchChatRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=64)


class BatchChatResponse(BaseModel):
    responses: List[ChatResponse]
    batch_size: int
    wall_time_ms: float
    requests_per_sec: float


class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)


class RouteResponse(BaseModel):
    category: str
    confidence: float
    scores: Dict[str, float]
    model_tier: str
    backend: str
    route_time_ms: float


class RouterBenchmarkRequest(BaseModel):
    num_intents: int = Field(2000, ge=100, le=200_000)


class BenchmarkRunRequest(BaseModel):
    num_prompts: int = Field(20, ge=1, le=200)
    modes: Optional[List[str]] = Field(
        None, description="Subset of modes to test; defaults to all."
    )


class ModeResult(BaseModel):
    mode: str
    num_requests: int
    wall_time_s: float
    requests_per_sec: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_ttft_ms: float
    avg_tokens_per_sec: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    est_cost_per_1m_tokens: float
    cache_hit_rate: Optional[float] = None
    notes: Optional[str] = None


class BenchmarkResult(BaseModel):
    id: Optional[int] = None
    created_at: str
    inference_mode: str
    num_prompts: int
    results: List[ModeResult]

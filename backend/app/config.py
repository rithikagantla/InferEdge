"""Central configuration for InferEdge.

Everything is driven by environment variables so the same image runs in
mock mode (no GPU, no API key) or against real NVIDIA NIM endpoints.
"""
import os
from dataclasses import dataclass
from typing import Dict

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "mock").lower()  # mock | nim | local
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_NIM_MODEL = os.getenv("NVIDIA_NIM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
NVIDIA_NIM_BASE_URL = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

DB_PATH = os.getenv("INFEREDGE_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "inferedge.db"))

# Rolling window used by GET /metrics aggregations.
METRICS_WINDOW = int(os.getenv("METRICS_WINDOW", "200"))

# Prompt cache
PROMPT_CACHE_MAX_ENTRIES = int(os.getenv("PROMPT_CACHE_MAX_ENTRIES", "512"))


@dataclass(frozen=True)
class ModelTier:
    """A model tier with pricing used for cost-per-token analysis.

    Prices are USD per 1M tokens, modeled on typical hosted-NIM pricing.
    They are estimates — swap in your real contract pricing here.
    """

    name: str
    input_cost_per_1m: float
    output_cost_per_1m: float


# "small" is what the semantic router sends easy intents to (e.g. an 8B
# model); "large" handles hard/technical intents (e.g. Nemotron 70B).
MODEL_TIERS: Dict[str, ModelTier] = {
    "small": ModelTier("meta/llama-3.1-8b-instruct (simulated)", 0.06, 0.24),
    "large": ModelTier(NVIDIA_NIM_MODEL, 0.35, 1.40),
    # Same weights as "large" but served through a (simulated) INT8/FP8
    # TensorRT-LLM engine: faster decode, same per-token API price.
    "quantized": ModelTier(NVIDIA_NIM_MODEL + " (INT8 simulated)", 0.35, 1.40),
}


def estimate_cost_usd(model_tier: str, input_tokens: int, output_tokens: int) -> float:
    tier = MODEL_TIERS.get(model_tier, MODEL_TIERS["large"])
    return (
        input_tokens * tier.input_cost_per_1m
        + output_tokens * tier.output_cost_per_1m
    ) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used when the provider does
    not return usage numbers (mock mode)."""
    return max(1, round(len(text) / 4))

"""Central configuration for InferEdge.

Everything is driven by environment variables so the same image runs in
mock mode (no GPU, no API key) or against real NVIDIA NIM endpoints.
"""
import os
from dataclasses import dataclass
from typing import Dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_dotenv(path: str) -> None:
    """Minimal .env loader (KEY=VALUE lines) so a local uvicorn run picks
    up the same file docker-compose uses. Real environment variables
    always win over the file."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ.setdefault(key.strip(), value)


_load_dotenv(os.path.join(_REPO_ROOT, ".env"))

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "mock").lower()  # mock | nim | local
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
# Hosted model availability changes: Llama 3.1 Nemotron 70B and Minitron 8B
# are still listed in the catalog but return 404 for new keys, so both tiers
# use current Nemotron mixture-of-experts models. Ultra activates ~55B of its
# 550B parameters per token; Lightning activates ~3B of 30B.
NVIDIA_NIM_MODEL = os.getenv("NVIDIA_NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
NVIDIA_NIM_SMALL_MODEL = os.getenv(
    "NVIDIA_NIM_SMALL_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
)
# Nemotron 3.x reasons before answering by default. A real-time support
# agent can't afford that: with a 256-token budget Lightning can spend every
# token reasoning and return no answer at all.
NIM_DISABLE_THINKING = os.getenv("NIM_DISABLE_THINKING", "true").lower() == "true"
NVIDIA_NIM_BASE_URL = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
# The free build.nvidia.com tier is rate limited, so cap in-flight requests
# and retry 429/5xx with exponential backoff.
NIM_MAX_CONCURRENCY = int(os.getenv("NIM_MAX_CONCURRENCY", "4"))
NIM_MAX_RETRIES = int(os.getenv("NIM_MAX_RETRIES", "5"))

# Self-hosted OpenAI-compatible server (vllm serve / trtllm-serve).
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "NousResearch/Meta-Llama-3-8B-Instruct")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "")
LOCAL_SMALL_BASE_URL = os.getenv("LOCAL_SMALL_BASE_URL", "")
LOCAL_SMALL_MODEL = os.getenv("LOCAL_SMALL_MODEL", "")
# A self-hosted GPU has no rate limit; the server batches everything sent.
LOCAL_MAX_CONCURRENCY = int(os.getenv("LOCAL_MAX_CONCURRENCY", "64"))

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
    "small": ModelTier(NVIDIA_NIM_SMALL_MODEL, 0.06, 0.24),
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

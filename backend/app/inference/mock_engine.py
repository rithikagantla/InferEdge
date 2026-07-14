"""Mock inference engine.

Produces realistic customer-support answers with simulated serving
behavior so the whole platform (metrics, benchmarks, dashboard) works
with zero external dependencies:

  * TTFT (time to first token) — fixed scheduling/prefill overhead
  * per-token decode time      — linear in output length
  * batching                   — requests in a batch run concurrently and
                                 share prefill overhead, mimicking
                                 continuous batching on a GPU
  * model tiers                — "small" decodes ~3x faster and costs
                                 ~6x less than "large", mimicking an 8B
                                 vs 70B deployment

Latency constants are tuned so a benchmark of 50-200 prompts finishes in
a demo-friendly amount of time while keeping realistic ratios.
"""
import asyncio
import math
import random
from typing import Dict, List

from .. import config
from .engine import InferenceEngine, InferenceResult

# Simulated serving profile per model tier: (ttft_seconds, seconds_per_token)
_TIER_PROFILE: Dict[str, tuple] = {
    "large": (0.080, 0.006),      # ~165 output tok/s — 70B-class on a modern GPU
    "small": (0.030, 0.002),      # ~500 output tok/s — 8B-class
    "quantized": (0.055, 0.0038), # ~260 tok/s — 70B with INT8/FP8 (simulated)
}

_RESPONSES: Dict[str, str] = {
    "billing": (
        "I'm sorry about the billing trouble. I've checked your account and I can see the "
        "charge you're referring to. If you were double-charged, the duplicate is "
        "automatically flagged and reversed within 3-5 business days. I've also sent a "
        "detailed invoice to your email. Is there anything else about your bill I can clarify?"
    ),
    "technical_support": (
        "Thanks for reporting this — let's get it fixed. First, please update to the latest "
        "app version and restart your device. If the issue persists, clear the app cache "
        "under Settings > Storage. I've logged a diagnostic ticket on your account so our "
        "engineers can see the crash reports. You'll get an email update within 24 hours."
    ),
    "account_management": (
        "I can help with your account. I've sent a secure password-reset link to the email "
        "on file — it expires in 30 minutes. For plan changes like upgrades, you can pick a "
        "new tier under Settings > Subscription and it takes effect immediately, prorated to "
        "your current billing cycle. Let me know if you'd like me to walk you through it."
    ),
    "refund": (
        "I understand you'd like a refund, and I'm happy to help. I've started a refund "
        "request for your most recent order. Once approved, the amount is returned to your "
        "original payment method within 5-7 business days. You'll receive a confirmation "
        "email with the refund ID shortly. Is there anything else I can do for you?"
    ),
    "shipping": (
        "Sorry your delivery is running late! I've checked the carrier feed: your package "
        "is currently at the regional distribution center and is expected within 2 business "
        "days. I've enabled priority handling and live tracking alerts on your order, so "
        "you'll get an SMS the moment it's out for delivery."
    ),
    "human_escalation": (
        "I completely understand, and I'm escalating this right away. I've created a "
        "priority ticket and assigned it to a senior support specialist. A human agent will "
        "contact you within 15 minutes during business hours. Your reference number is "
        "SUP-20260713. Thank you for your patience — we'll make this right."
    ),
    "general": (
        "Thanks for reaching out to support! I've reviewed your request and here's what I "
        "found: everything on your account looks healthy, and I've noted your question for "
        "our team. Could you share a little more detail so I can give you a precise answer? "
        "I'm happy to help with billing, orders, technical issues, or account changes."
    ),
}


def _pick_category(prompt: str) -> str:
    """Lightweight keyword match so mock answers are on-topic.

    The real classification for routing decisions is done by the semantic
    router (app/routing) — this only selects the canned response text.
    """
    p = prompt.lower()
    checks = [
        ("human_escalation", ("human", "agent", "person", "manager", "escalate", "complaint")),
        ("refund", ("refund", "money back", "return")),
        ("billing", ("charge", "bill", "invoice", "payment", "price", "subscription price")),
        ("shipping", ("order", "package", "deliver", "ship", "tracking")),
        ("account_management", ("password", "log in", "login", "account", "upgrade", "email", "subscription")),
        ("technical_support", ("crash", "error", "bug", "app", "sync", "timeout", "lag")),
    ]
    for category, keywords in checks:
        if any(k in p for k in keywords):
            return category
    return "general"


class MockEngine(InferenceEngine):
    name = "mock"

    async def generate(self, prompt: str, model_tier: str = "large") -> InferenceResult:
        return (await self._generate_shared(prompt, model_tier, batch_size=1))

    async def generate_batch(
        self, prompts: List[str], model_tier: str = "large"
    ) -> List[InferenceResult]:
        # Continuous batching simulation: all requests decode concurrently.
        # Per-token time degrades mildly with batch size (log2 factor), but
        # wall time collapses from sum(latencies) to ~max(latencies), which
        # is exactly the throughput win real GPU batching delivers.
        batch_size = len(prompts)
        return list(
            await asyncio.gather(
                *(self._generate_shared(p, model_tier, batch_size) for p in prompts)
            )
        )

    async def _generate_shared(
        self, prompt: str, model_tier: str, batch_size: int
    ) -> InferenceResult:
        ttft_s, per_token_s = _TIER_PROFILE.get(model_tier, _TIER_PROFILE["large"])
        category = _pick_category(prompt)
        text = _RESPONSES[category]

        input_tokens = config.estimate_tokens(prompt)
        output_tokens = config.estimate_tokens(text)

        # Mild per-token slowdown as the batch grows (kernel contention),
        # jitter so percentile metrics behave like real traffic.
        contention = 1.0 + 0.08 * math.log2(max(batch_size, 1))
        jitter = random.uniform(0.9, 1.25)
        ttft = ttft_s * jitter
        decode = output_tokens * per_token_s * contention * jitter

        await asyncio.sleep(ttft + decode)

        tier_name = config.MODEL_TIERS[model_tier].name if model_tier in config.MODEL_TIERS else model_tier
        return InferenceResult(
            text=text,
            model="mock/" + tier_name,
            model_tier=model_tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft * 1000,
            latency_ms=(ttft + decode) * 1000,
        )

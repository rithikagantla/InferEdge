"""API tests — run against the mock engine (no GPU / API key needed)."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["INFERENCE_MODE"] = "mock"
os.environ["INFEREDGE_DB_PATH"] = os.path.join(
    os.path.dirname(__file__), "test_inferedge.db"
)

from app.main import app  # noqa: E402
from app import service  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    db_path = os.environ["INFEREDGE_DB_PATH"]
    if os.path.exists(db_path):
        os.remove(db_path)


def test_root(client):
    data = client.get("/").json()
    assert data["service"] == "InferEdge"
    assert data["inference_mode"] == "mock"


def test_chat_baseline(client):
    resp = client.post("/chat", json={"prompt": "My order is delayed. Can you help?"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["response"]) > 20
    m = data["metrics"]
    assert m["mode"] == "baseline"
    assert m["latency_ms"] > 0
    assert m["input_tokens"] > 0 and m["output_tokens"] > 0
    assert m["tokens_per_sec"] > 0
    assert m["cost_usd"] > 0


def test_chat_rejects_bad_mode(client):
    resp = client.post("/chat", json={"prompt": "hi there", "mode": "warp-speed"})
    assert resp.status_code == 422


def test_chat_rejects_empty_prompt(client):
    resp = client.post("/chat", json={"prompt": ""})
    assert resp.status_code == 422


def test_caching_hit_is_faster_and_free(client):
    service.prompt_cache.clear()
    prompt = "I was charged twice this month."
    first = client.post("/chat", json={"prompt": prompt, "mode": "caching"}).json()
    second = client.post("/chat", json={"prompt": prompt, "mode": "caching"}).json()
    assert first["metrics"]["cached"] is False
    assert second["metrics"]["cached"] is True
    assert second["metrics"]["cost_usd"] == 0.0
    assert second["metrics"]["latency_ms"] < first["metrics"]["latency_ms"] / 10


def test_routing_mode_classifies(client):
    resp = client.post(
        "/chat", json={"prompt": "I want a refund for my last order.", "mode": "routing"}
    ).json()
    assert resp["metrics"]["category"] == "refund"
    assert resp["metrics"]["router_time_ms"] is not None


def test_batch_chat_throughput_beats_sequential(client):
    prompts = [
        "My order is delayed. Can you help?",
        "I was charged twice this month.",
        "The app keeps crashing after the latest update.",
        "Can I upgrade my subscription?",
    ]
    resp = client.post("/batch-chat", json={"prompts": prompts})
    assert resp.status_code == 200
    data = resp.json()
    assert data["batch_size"] == 4
    assert len(data["responses"]) == 4
    # Concurrent batch: wall time must be far below the sum of latencies.
    total_latency = sum(r["metrics"]["latency_ms"] for r in data["responses"])
    assert data["wall_time_ms"] < total_latency * 0.6


def test_route_endpoint(client):
    resp = client.post("/route", json={"prompt": "I forgot my password and cannot log in."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "account_management"
    assert data["model_tier"] == "small"
    assert set(data["scores"].keys()) == {
        "billing", "technical_support", "account_management",
        "refund", "shipping", "human_escalation",
    }


def test_route_all_categories():
    """Router sanity check across representative phrasings."""
    from app.routing import get_router

    expectations = {
        "Why did my card get billed two times?": "billing",
        "The checkout page throws an error every time": "technical_support",
        "Please reset my password, I am locked out": "account_management",
        "I would like my money back for this order": "refund",
        "My package tracking shows no movement": "shipping",
        "Get me a real human on the line now": "human_escalation",
    }
    router = get_router()
    for prompt, expected in expectations.items():
        assert router.classify(prompt)["category"] == expected, prompt


def test_router_benchmark(client):
    resp = client.post("/router/benchmark", json={"num_intents": 1000})
    assert resp.status_code == 200
    data = resp.json()
    assert data["cpu_time_ms"] > 0
    assert data["num_intents"] == 1000
    # On a machine without an NVIDIA GPU we must be in CPU fallback mode.
    if not data["gpu_available"]:
        assert data["gpu_time_ms"] is None
        assert "CPU fallback" in data["note"]


def test_metrics_endpoint(client):
    data = client.get("/metrics").json()
    assert data["total_requests"] > 0
    overall = data["overall"]
    assert overall["p95_latency_ms"] >= overall["p50_latency_ms"]
    assert overall["est_cost_per_1m_tokens"] >= 0
    assert "baseline" in data["by_mode"]


def test_benchmark_run_and_results(client):
    resp = client.post(
        "/benchmark/run",
        json={"num_prompts": 6, "modes": ["baseline", "batching", "caching", "routing", "quantized"]},
    )
    assert resp.status_code == 200
    run = resp.json()
    modes = {r["mode"]: r for r in run["results"]}
    assert set(modes) == {"baseline", "batching", "caching", "routing", "quantized"}
    # Batching must beat sequential baseline on throughput.
    assert modes["batching"]["requests_per_sec"] > modes["baseline"]["requests_per_sec"]
    # Routing must be cheaper per token than baseline (small-model traffic).
    assert modes["routing"]["est_cost_per_1m_tokens"] < modes["baseline"]["est_cost_per_1m_tokens"]
    # Quantized (simulated) must have lower p50 than baseline.
    assert modes["quantized"]["p50_latency_ms"] < modes["baseline"]["p50_latency_ms"]

    saved = client.get("/benchmark-results").json()
    assert saved["runs"][0]["id"] == run["id"]
    assert saved["runs"][0]["num_prompts"] == 6


def test_sample_prompts(client):
    data = client.get("/sample-prompts").json()
    assert len(data["prompts"]) >= 20

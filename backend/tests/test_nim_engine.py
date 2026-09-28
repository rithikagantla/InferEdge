"""NIM engine tests against a fake OpenAI-compatible SSE endpoint.

No network or API key needed: httpx.MockTransport stands in for NIM.
"""
import asyncio
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import config  # noqa: E402
from app.inference import nim_engine  # noqa: E402
from app.inference.nim_engine import NIMEngine, NIMError  # noqa: E402


def sse(*chunks, done=True) -> bytes:
    lines = ["data: " + json.dumps(c) for c in chunks]
    if done:
        lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def delta(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


ROLE_CHUNK = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
USAGE_CHUNK = {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 7}}


@pytest.fixture(autouse=True)
def nim_config(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setattr(nim_engine, "_BACKOFF_BASE_S", 0.001)


def make_engine(handler):
    return NIMEngine(transport=httpx.MockTransport(handler))


def test_streams_text_usage_and_ttft():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(
            200, content=sse(ROLE_CHUNK, delta("Hello"), delta(", world"), USAGE_CHUNK)
        )

    result = asyncio.run(make_engine(handler).generate("hi", "large"))
    assert result.text == "Hello, world"
    assert result.input_tokens == 42 and result.output_tokens == 7
    assert 0 < result.ttft_ms <= result.latency_ms
    assert result.model == config.NVIDIA_NIM_MODEL
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert seen["auth"] == "Bearer nvapi-test"


def test_small_tier_uses_small_model():
    models = []

    def handler(request):
        models.append(json.loads(request.content)["model"])
        return httpx.Response(200, content=sse(delta("ok")))

    engine = make_engine(handler)
    asyncio.run(engine.generate("hi", "small"))
    asyncio.run(engine.generate("hi", "quantized"))
    assert models == [config.NVIDIA_NIM_SMALL_MODEL, config.NVIDIA_NIM_MODEL]


def test_retries_rate_limit_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"}, text="rate limited")
        return httpx.Response(200, content=sse(delta("done")))

    result = asyncio.run(make_engine(handler).generate("hi"))
    assert result.text == "done"
    assert calls["n"] == 3


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(config, "NIM_MAX_RETRIES", 2)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="overloaded")

    with pytest.raises(NIMError, match="after 2 retries"):
        asyncio.run(make_engine(handler).generate("hi"))
    assert calls["n"] == 3


def test_falls_back_when_stream_options_rejected():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "stream_options" in body:
            return httpx.Response(400, text="Unrecognized field: stream_options")
        return httpx.Response(200, content=sse(delta("a reply without usage")))

    engine = make_engine(handler)
    result = asyncio.run(engine.generate("hi"))
    assert result.text == "a reply without usage"
    assert result.output_tokens == config.estimate_tokens("a reply without usage")
    assert "stream_options" not in bodies[-1]
    # Remembered: later requests skip stream_options up front.
    asyncio.run(engine.generate("again"))
    assert len(bodies) == 3


def test_auth_and_missing_model_errors_are_not_retried():
    for status, needle in ((401, "NVIDIA_API_KEY"), (404, "removed from the hosted catalog")):
        calls = {"n": 0}

        def handler(request, status=status):
            calls["n"] += 1
            return httpx.Response(status, text="nope")

        with pytest.raises(NIMError, match=needle):
            asyncio.run(make_engine(handler).generate("hi"))
        assert calls["n"] == 1


def test_concurrency_is_capped(monkeypatch):
    monkeypatch.setattr(config, "NIM_MAX_CONCURRENCY", 2)
    state = {"active": 0, "peak": 0}

    async def handler(request):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.02)
        state["active"] -= 1
        return httpx.Response(200, content=sse(delta("ok")))

    engine = make_engine(handler)
    results = asyncio.run(engine.generate_batch(["p%d" % i for i in range(6)]))
    assert len(results) == 6
    assert state["peak"] == 2


def test_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        NIMEngine()

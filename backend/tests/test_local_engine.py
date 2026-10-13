"""LocalEngine (self-hosted vLLM / TensorRT-LLM) against a fake server."""
import asyncio
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import config  # noqa: E402
from app.inference.local_engine import LocalEngine  # noqa: E402


def test_requires_base_url(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_BASE_URL", "")
    with pytest.raises(RuntimeError, match="LOCAL_BASE_URL"):
        LocalEngine()


def test_generates_against_local_server(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_BASE_URL", "http://gpu-box/v1")
    monkeypatch.setattr(config, "LOCAL_MODEL", "my-model")
    monkeypatch.setattr(config, "LOCAL_API_KEY", "")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["model"] = json.loads(request.content)["model"]
        seen["auth"] = request.headers.get("authorization")
        chunks = [
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 1}},
        ]
        body = "".join("data: %s\n\n" % json.dumps(c) for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode())

    engine = LocalEngine(transport=httpx.MockTransport(handler))
    # Without a second server, routed small-tier traffic uses the one model.
    result = asyncio.run(engine.generate("hello", "small"))
    assert result.text == "Hi"
    assert result.input_tokens == 12 and result.output_tokens == 1
    assert seen["url"] == "http://gpu-box/v1/chat/completions"
    assert seen["model"] == "my-model"
    assert seen["auth"] is None

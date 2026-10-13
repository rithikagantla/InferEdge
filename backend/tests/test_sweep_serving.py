"""Tests for scripts/sweep_serving.py against a fake streaming server
whose per-chunk timing is controlled, so TTFT/TPOT/ITL math is checkable."""
import asyncio
import importlib.util
import json
import os
import sys

import httpx
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.inference.openai_compat import StreamResult  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sweep_serving", os.path.join(ROOT, "scripts", "sweep_serving.py")
)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def test_percentile_matches_numpy_linear():
    vals = [10.0, 20.0, 30.0, 40.0]
    assert sweep.percentile(vals, 50) == 25.0
    assert sweep.percentile(vals, 0) == 10.0
    assert sweep.percentile(vals, 100) == 40.0
    assert sweep.percentile([5.0], 99) == 5.0
    assert sweep.percentile([], 50) is None


def test_request_stats_tpot_and_itl():
    result = StreamResult(
        text="abcd", ttft_ms=100.0, latency_ms=160.0,
        chunk_times_ms=[100.0, 120.0, 140.0, 160.0], completion_tokens=4,
    )
    stats = sweep.request_stats(result)
    assert stats["tpot_ms"] == pytest.approx(20.0)  # (160 - 100) / (4 - 1)
    assert stats["itls_ms"] == pytest.approx([20.0, 20.0, 20.0])
    assert stats["output_tokens"] == 4


def test_request_stats_falls_back_to_chunk_count():
    result = StreamResult(text="ab", ttft_ms=50.0, latency_ms=60.0, chunk_times_ms=[50.0, 60.0])
    assert sweep.request_stats(result)["output_tokens"] == 2


def fake_server(tokens=5, ttft_s=0.03, gap_s=0.01, reject_stream_options=False):
    """Async MockTransport that streams `tokens` chunks with fixed timing
    and tracks peak in-flight requests."""
    state = {"active": 0, "peak": 0, "bodies": []}

    async def handler(request):
        body = json.loads(request.content)
        state["bodies"].append(body)
        if reject_stream_options and "stream_options" in body:
            return httpx.Response(400, text="unknown field stream_options")

        async def stream():
            # The client stops reading at [DONE], so count the request as
            # finished just before sending it rather than in a finally.
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(ttft_s)
            for i in range(tokens):
                if i:
                    await asyncio.sleep(gap_s)
                chunk = {"choices": [{"delta": {"content": "t%d " % i}}]}
                yield ("data: %s\n\n" % json.dumps(chunk)).encode()
            if "stream_options" in body:
                usage = {"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": tokens}}
                yield ("data: %s\n\n" % json.dumps(usage)).encode()
            state["active"] -= 1
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, content=stream())

    return httpx.MockTransport(handler), state


def test_run_sweep_measures_levels():
    transport, state = fake_server(tokens=5, ttft_s=0.03, gap_s=0.01)
    result = asyncio.run(sweep.run_sweep(
        "http://fake/v1", "m", "test", levels=[1, 4], min_requests=4, per_worker=1,
        warmup=1, ignore_eos=True, log=lambda *_: None, transport=transport,
    ))
    one, four = result["levels"]
    assert one["requests"] == 4 and four["requests"] == 4
    assert one["errors"] == 0 and four["errors"] == 0
    assert state["peak"] == 4
    assert one["avg_output_tokens"] == 5
    # Fake server: TTFT ~30 ms, ~10 ms between tokens.
    assert 25 <= one["ttft_p50_ms"] < 80
    assert 8 <= one["tpot_p50_ms"] < 30
    assert 8 <= one["itl_p50_ms"] < 30
    # 4 concurrent streams finish ~4x faster than 4 sequential ones.
    assert four["requests_per_sec"] > 2 * one["requests_per_sec"]
    assert all(b["ignore_eos"] is True and b["stream"] is True for b in state["bodies"])
    assert result["stream_usage_reported"] is True


def test_run_sweep_handles_stream_options_rejection():
    transport, state = fake_server(tokens=3, ttft_s=0.0, gap_s=0.0, reject_stream_options=True)
    result = asyncio.run(sweep.run_sweep(
        "http://fake/v1", "m", "test", levels=[2], min_requests=2, per_worker=1,
        warmup=1, log=lambda *_: None, transport=transport,
    ))
    level = result["levels"][0]
    assert level["errors"] == 0 and level["requests"] == 2
    assert level["avg_output_tokens"] == 3  # counted from chunks, no usage
    assert result["stream_usage_reported"] is False
    assert "ignore_eos" not in state["bodies"][-1]


def test_errors_are_counted_not_raised():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:  # warm-up succeeds
            return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(503, text="overloaded")

    result = asyncio.run(sweep.run_sweep(
        "http://fake/v1", "m", "test", levels=[2], min_requests=3, per_worker=1,
        warmup=1, log=lambda *_: None, transport=httpx.MockTransport(handler),
    ))
    level = result["levels"][0]
    assert level["errors"] == 3 and level["requests"] == 0
    assert "503" in level["first_error"]


def test_write_outputs(tmp_path):
    data = {
        "label": "unit", "levels": [sweep.summarize_level(1, [
            {"ttft_ms": 10.0, "e2e_ms": 50.0, "output_tokens": 5, "tpot_ms": 10.0, "itls_ms": [10.0]}
        ], 0, 0.05)],
    }
    base = sweep.write_outputs(data, str(tmp_path))
    for ext in (".json", ".csv", ".md"):
        assert os.path.exists(base + ext)
    assert "| 1 |" in open(base + ".md").read()

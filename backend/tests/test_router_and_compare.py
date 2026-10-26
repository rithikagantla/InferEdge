"""Router batch classification, the held-out eval set, and route_compare's report math."""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.router_eval_set import ROUTER_EVAL_SET  # noqa: E402
from app.routing import semantic_router as sr  # noqa: E402
from app.routing.semantic_router import CATEGORY_EXEMPLARS, SemanticRouter  # noqa: E402
from app.sample_prompts import SAMPLE_PROMPTS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "route_compare", os.path.join(ROOT, "scripts", "route_compare.py"))
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def test_eval_set_is_held_out_and_balanced():
    exemplars = {p.lower() for ps in CATEGORY_EXEMPLARS.values() for p in ps}
    samples = {p.lower() for p in SAMPLE_PROMPTS}
    for text, category in ROUTER_EVAL_SET:
        assert text.lower() not in exemplars and text.lower() not in samples
        assert category in CATEGORY_EXEMPLARS
    counts = {c: sum(1 for _, y in ROUTER_EVAL_SET if y == c) for c in CATEGORY_EXEMPLARS}
    assert set(counts.values()) == {10}


def test_classify_batch_matches_classify():
    router = SemanticRouter("hashed")
    texts = [t for t, _ in ROUTER_EVAL_SET[:12]]
    assert router.classify_batch(texts) == [router.classify(t)["category"] for t in texts]


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None,
                    reason="sentence-transformers not installed")
def test_minilm_router_runs():  # pragma: no cover - optional dependency
    router = SemanticRouter("minilm")
    assert router.classify("Transfer me to a supervisor please.")["category"] in CATEGORY_EXEMPLARS


def test_workload_labels_follow_prompt_blocks():
    items = rc.workload()
    assert len(items) == 50
    assert items[0]["category"] == "shipping" and items[25]["category"] == "human_escalation"
    assert items[30]["prompt"] == SAMPLE_PROMPTS[0]  # cycles after 30
    assert {i["expected_tier"] for i in items if i["category"] in rc.HARD} == {"large"}


def test_report_counts_judge_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(rc, "RESULTS", str(tmp_path / "r.jsonl"))
    rows = []
    for strategy, tier in (("always-ultra", "large"), ("sy-classifier", "small")):
        for i, item in enumerate(rc.workload()[:4]):
            rows.append(dict(item, strategy=strategy, tier=tier,
                             served_model=rc.STRONG if tier == "large" else rc.WEAK,
                             ttft_ms=100.0 + i, latency_ms=500.0, prompt_tokens=50,
                             completion_tokens=100, text="ok", error=None, score=4))
    rows.append({"summary": True, "strategy": "sy-classifier", "wall_s": 1.0,
                 "judge": {"calls": 4, "prompt_tokens": 4000, "completion_tokens": 400}})
    with open(rc.RESULTS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rc.report()
    out = {r["strategy"]: r for r in json.load(open(tmp_path / "route_compare.json"))}
    assert out["always-ultra"]["weak_share"] == 0 and out["sy-classifier"]["weak_share"] == 1
    assert out["sy-classifier"]["judge_cost_usd"] > 0
    assert out["sy-classifier"]["total_cost_usd"] == pytest.approx(
        out["sy-classifier"]["answer_cost_usd"] + out["sy-classifier"]["judge_cost_usd"], abs=1e-6)
    assert out["always-ultra"]["quality_mean"] == 4

#!/usr/bin/env python3
"""Compare routing strategies on the 50-prompt support benchmark.

Every strategy answers the same prompts with the same system prompt and
sampling settings, choosing between the same two hosted NIM models:
Nemotron 3 Ultra (strong) and Nemotron 3.5 Lightning (weak).

  always-ultra        every request to the strong model (quality ceiling)
  always-lightning    every request to the weak model (cost floor)
  inferedge-semantic  InferEdge's embedding router picks the tier (no LLM call)
  sy-random           NVIDIA NeMo Switchyard, random 50/50 split
  sy-classifier       Switchyard LLM classifier: a judge model estimates whether
                      the weak model can handle the request
  sy-random-matched   Switchyard random split weighted to the classifier's
                      observed weak share (same budget, no judgement)

Answers are then graded blind (1-5) by a model that is not one of the two
being routed between. Usage:

    python scripts/route_compare.py run --strategies always-ultra,sy-classifier
    python scripts/route_compare.py grade
    python scripts/route_compare.py report

Switchyard must be running for sy-* strategies (see switchyard/nim_routes.toml).
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import httpx  # noqa: E402

from app import config  # noqa: E402
from app.inference.openai_compat import (  # noqa: E402
    SYSTEM_PROMPT,
    InferenceAPIError,
    RetryableError,
    stream_chat_completion,
)
from app.routing import get_router  # noqa: E402
from app.sample_prompts import SAMPLE_PROMPTS  # noqa: E402

OUT_DIR = "benchmark_results/switchyard"
RESULTS = os.path.join(OUT_DIR, "route_compare.jsonl")
STRONG = config.NVIDIA_NIM_MODEL
WEAK = config.NVIDIA_NIM_SMALL_MODEL
GRADER = "openai/gpt-oss-20b"
SWITCHYARD_URL = os.getenv("SWITCHYARD_URL", "http://127.0.0.1:4000/v1")
NUM_PROMPTS = 50

# SAMPLE_PROMPTS is ordered in blocks of five per category.
CATEGORIES = ["shipping", "billing", "account_management", "refund",
              "technical_support", "human_escalation"]
# InferEdge's policy: these need the strong model; the rest are routine.
HARD = {"technical_support", "human_escalation"}

THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

GRADER_PROMPT = """You are grading a customer-support assistant's reply.
The assistant has no access to the customer's account, orders or systems.
Reward replies that are accurate, specific and actionable, take the right next
step (including escalating to a human when the customer asks for one), and are
honest about what the assistant cannot see. Penalize invented account details or
claimed actions, generic filler, wrong advice, and ignoring the request.

Customer message:
{prompt}

Assistant reply:
{reply}

Score 1-5: 1 = harmful or useless, 2 = poor, 3 = acceptable, 4 = good,
5 = excellent. Respond with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def workload() -> List[dict]:
    items = []
    for i in range(NUM_PROMPTS):
        j = i % len(SAMPLE_PROMPTS)
        category = CATEGORIES[j // 5]
        items.append({"idx": i, "prompt": SAMPLE_PROMPTS[j], "category": category,
                      "expected_tier": "large" if category in HARD else "small"})
    return items


def tier_of(model: Optional[str]) -> Optional[str]:
    if model == STRONG:
        return "large"
    if model == WEAK:
        return "small"
    return None


def nim_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=config.NVIDIA_NIM_BASE_URL, timeout=httpx.Timeout(180, connect=15),
                             headers={"Authorization": "Bearer " + config.NVIDIA_API_KEY})


def messages(prompt: str) -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


async def call(client: httpx.AsyncClient, payload: dict, retries: int = 4):
    """One request with retry on rate limits / transient errors."""
    for attempt in range(retries + 1):
        try:
            return await stream_chat_completion(client, payload)
        except RetryableError as exc:
            if attempt == retries:
                raise
            await asyncio.sleep(exc.retry_after if exc.retry_after is not None else 2 * 2 ** attempt)


async def answer(strategy: str, item: dict, nim: httpx.AsyncClient, sy: httpx.AsyncClient) -> dict:
    base = {"model": None, "messages": messages(item["prompt"]), "max_tokens": 256,
            "temperature": 0.3, "stream_options": {"include_usage": True}}
    if strategy.startswith("sy-"):
        payload = dict(base, model=strategy)
        client = sy
    else:
        if strategy == "always-ultra":
            model = STRONG
        elif strategy == "always-lightning":
            model = WEAK
        elif strategy == "inferedge-semantic":
            route = get_router().classify(item["prompt"])
            model = WEAK if route["model_tier"] == "small" else STRONG
            item = dict(item, router_category=route["category"])
        else:
            raise ValueError(strategy)
        payload = dict(base, model=model, **THINKING_OFF)
        client = nim
    t0 = time.time()
    try:
        r = await call(client, payload)
        served = r.headers.get("x-model-router-selected-model") or payload["model"]
        return dict(item, strategy=strategy, served_model=served, tier=tier_of(served),
                    ttft_ms=r.ttft_ms, latency_ms=r.latency_ms, prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens, text=r.text, error=None, ts=t0)
    except (RetryableError, InferenceAPIError, ValueError) as exc:
        return dict(item, strategy=strategy, served_model=None, tier=None, ttft_ms=None,
                    latency_ms=None, text="", error=str(exc)[:300], ts=t0)


def sy_stats() -> Optional[dict]:
    try:
        r = httpx.get(SWITCHYARD_URL.rsplit("/v1", 1)[0] + "/v1/stats", timeout=10)
        return r.json()
    except Exception:
        return None


async def run(strategies: List[str], concurrency: int) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    items = workload()
    async with nim_client() as nim, httpx.AsyncClient(base_url=SWITCHYARD_URL,
                                                      timeout=httpx.Timeout(300, connect=15)) as sy:
        for strategy in strategies:
            if strategy.startswith("sy-"):
                httpx.post(SWITCHYARD_URL.rsplit("/v1", 1)[0] + "/v1/stats/reset", timeout=10)
            sem = asyncio.Semaphore(concurrency)

            async def one(item):
                async with sem:
                    return await answer(strategy, item, nim, sy)

            t0 = time.perf_counter()
            rows = await asyncio.gather(*(one(it) for it in items))
            wall = time.perf_counter() - t0
            judge = None
            if strategy.startswith("sy-"):
                stats = sy_stats() or {}
                clf = stats.get("classifier") or {}
                tokens = clf.get("total_tokens") or {}
                lat = ((clf.get("models") or {}).get(GRADER) or {}).get("model_call_latency") or {}
                judge = {"calls": clf.get("total_requests", 0), "prompt_tokens": tokens.get("prompt", 0),
                         "completion_tokens": tokens.get("completion", 0), "avg_ms": lat.get("avg_ms"),
                         "p50_ms": lat.get("p50_ms")}
            with open(RESULTS, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                f.write(json.dumps({"summary": True, "strategy": strategy, "wall_s": wall,
                                    "judge": judge}) + "\n")
            ok = [r for r in rows if not r["error"]]
            small = sum(1 for r in ok if r["tier"] == "small")
            print("%-19s %2d ok, %2d errors, %2d to %s, %.0fs" % (
                strategy, len(ok), len(rows) - len(ok), small, WEAK.split("/")[1], wall), flush=True)


def load() -> List[dict]:
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS) as f:
        return [json.loads(line) for line in f if line.strip()]


async def grade(concurrency: int) -> None:
    rows = load()
    todo = [r for r in rows if not r.get("summary") and not r.get("error") and r.get("score") is None]
    sem = asyncio.Semaphore(concurrency)
    async with nim_client() as nim:
        async def one(r):
            async with sem:
                # Room for the grader's reasoning plus the JSON verdict.
                payload = {"model": GRADER, "max_tokens": 1500, "temperature": 0.0,
                           "reasoning_effort": "medium",
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "user", "content": GRADER_PROMPT.format(
                               prompt=r["prompt"], reply=r["text"])}]}
                try:
                    res = await call(nim, payload)
                    m = re.search(r"\{.*\}", res.text, re.S)
                    verdict = json.loads(m.group(0)) if m else {}
                    r["score"] = int(verdict.get("score")) if verdict.get("score") is not None else None
                    r["grade_reason"] = verdict.get("reason")
                    if r["score"] is None:
                        r["grade_error"] = "unparseable verdict: %r" % res.text[:120]
                except Exception as exc:  # grading failure leaves the row ungraded
                    r["score"] = None
                    r["grade_error"] = str(exc)[:200]

        await asyncio.gather(*(one(r) for r in todo))
    with open(RESULTS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("graded %d answers" % sum(1 for r in todo if r.get("score") is not None))


def pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def report() -> None:
    rows = load()
    order = ["always-ultra", "always-lightning", "inferedge-semantic", "sy-classifier",
             "sy-random", "sy-random-matched"]
    summaries = {r["strategy"]: r for r in rows if r.get("summary")}
    out = []
    for s in order:
        rs = [r for r in rows if not r.get("summary") and r["strategy"] == s]
        if not rs:
            continue
        ok = [r for r in rs if not r["error"]]
        scored = [r for r in ok if r.get("score") is not None]
        hard = [r for r in scored if r["category"] in HARD]
        easy = [r for r in scored if r["category"] not in HARD]
        cost = 0.0
        tokens = 0
        for r in ok:
            tier = r["tier"] or "large"
            pt = r.get("prompt_tokens") or config.estimate_tokens(SYSTEM_PROMPT + r["prompt"])
            ct = r.get("completion_tokens") or config.estimate_tokens(r["text"])
            cost += config.estimate_cost_usd(tier, pt, ct)
            tokens += pt + ct
        judge = (summaries.get(s) or {}).get("judge") or {}
        judge_cost = config.estimate_cost_usd(
            "small", judge.get("prompt_tokens", 0) or 0, judge.get("completion_tokens", 0) or 0)
        agree = [r for r in ok if r["tier"] == r["expected_tier"]]
        row = {
            "strategy": s,
            "requests": len(rs),
            "errors": len(rs) - len(ok),
            "weak_share": round(sum(1 for r in ok if r["tier"] == "small") / len(ok), 3) if ok else None,
            "policy_agreement": round(len(agree) / len(ok), 3) if ok else None,
            "quality_mean": round(statistics.mean(r["score"] for r in scored), 2) if scored else None,
            "quality_hard": round(statistics.mean(r["score"] for r in hard), 2) if hard else None,
            "quality_easy": round(statistics.mean(r["score"] for r in easy), 2) if easy else None,
            "good_or_better": round(sum(1 for r in scored if r["score"] >= 4) / len(scored), 3) if scored else None,
            "ttft_p50_ms": round(pct([r["ttft_ms"] for r in ok], 50), 0) if ok else None,
            "ttft_p95_ms": round(pct([r["ttft_ms"] for r in ok], 95), 0) if ok else None,
            "e2e_p50_ms": round(pct([r["latency_ms"] for r in ok], 50), 0) if ok else None,
            "answer_cost_usd": round(cost, 6),
            "judge_cost_usd": round(judge_cost, 6),
            "total_cost_usd": round(cost + judge_cost, 6),
            "judge_calls": judge.get("calls"),
            "judge_tokens": (judge.get("prompt_tokens") or 0) + (judge.get("completion_tokens") or 0),
            "judge_p50_ms": judge.get("p50_ms"),
        }
        out.append(row)
    base = next((r for r in out if r["strategy"] == "always-ultra"), None)
    lines = ["| Strategy | Weak-model share | Quality (1-5) | Quality, hard requests | Good or better | "
             "TTFT p50 (ms) | TTFT p95 (ms) | Cost incl. judge ($) | vs always-Ultra |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out:
        rel = ("%+.0f%%" % ((r["total_cost_usd"] / base["total_cost_usd"] - 1) * 100)
               if base and base["total_cost_usd"] else "—")
        lines.append("| %s | %.0f%% | %s | %s | %s | %s | %s | %.5f | %s |" % (
            r["strategy"], 100 * (r["weak_share"] or 0),
            r["quality_mean"], r["quality_hard"],
            "%.0f%%" % (100 * r["good_or_better"]) if r["good_or_better"] is not None else "—",
            "%.0f" % r["ttft_p50_ms"] if r["ttft_p50_ms"] is not None else "—",
            "%.0f" % r["ttft_p95_ms"] if r["ttft_p95_ms"] is not None else "—",
            r["total_cost_usd"], rel))
    table = "\n".join(lines) + "\n"
    # Same prompts for every strategy, so compare prompt by prompt against
    # always-ultra and bootstrap a 95% interval for the mean difference.
    import random
    rng = random.Random(0)
    scores = {}
    for r in rows:
        if not r.get("summary") and r.get("score") is not None:
            scores.setdefault(r["strategy"], {})[r["idx"]] = r["score"]
    ci_lines = ["", "Quality vs always-ultra, paired by prompt (95% bootstrap interval):", ""]
    for r in out:
        s = r["strategy"]
        if s == "always-ultra" or s not in scores or "always-ultra" not in scores:
            continue
        idx = sorted(set(scores[s]) & set(scores["always-ultra"]))
        d = [scores[s][i] - scores["always-ultra"][i] for i in idx]
        boots = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(5000))
        lo, hi = boots[125], boots[4874]
        r["quality_vs_ultra"] = {"mean": round(statistics.mean(d), 2), "ci95": [round(lo, 2), round(hi, 2)]}
        ci_lines.append("- %s: %+.2f [%+.2f, %+.2f]%s" % (
            s, statistics.mean(d), lo, hi, "" if lo > 0 or hi < 0 else " (not significant)"))
    table += "\n".join(ci_lines) + "\n"
    with open(os.path.join(OUT_DIR, "route_compare.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, "route_compare.md"), "w") as f:
        f.write(table)
    print(table)
    for r in out:
        print("%-19s policy agreement %.0f%% | easy %s hard %s | errors %d | judge calls %s, tokens %s, p50 %s ms" % (
            r["strategy"], 100 * (r["policy_agreement"] or 0), r["quality_easy"], r["quality_hard"],
            r["errors"], r["judge_calls"], r["judge_tokens"], r["judge_p50_ms"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--strategies", required=True)
    p_run.add_argument("--concurrency", type=int, default=2)
    p_grade = sub.add_parser("grade")
    p_grade.add_argument("--concurrency", type=int, default=3)
    sub.add_parser("report")
    args = parser.parse_args()
    if not config.NVIDIA_API_KEY:
        print("NVIDIA_API_KEY is required (put it in .env)", file=sys.stderr)
        return 1
    if args.cmd == "run":
        asyncio.run(run([s.strip() for s in args.strategies.split(",") if s.strip()], args.concurrency))
    elif args.cmd == "grade":
        asyncio.run(grade(args.concurrency))
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())

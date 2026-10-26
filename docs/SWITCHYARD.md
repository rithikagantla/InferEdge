# NVIDIA NeMo Switchyard in front of InferEdge's NIM models

[NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) is NVIDIA's open-source (Apache-2.0) LLM routing proxy. It exposes an OpenAI-compatible endpoint, picks a model for each request with a configurable algorithm, and forwards the request to that model. I put it in front of the same two hosted NIM models InferEdge routes between and compared its routing strategies with InferEdge's own semantic router on the 50-prompt support benchmark.

## Setup

- **Server:** `switchyard-server` 0.3.0, built from crates.io with `cargo install --locked switchyard-server --version 0.3.0`. The Python package (`nemo-switchyard`) is a library and doesn't include the server.
- **Config:** [`switchyard/nim_routes.toml`](../switchyard/nim_routes.toml). Three routes over one NIM client:
  - `sy-classifier`: `type = "llm_classifier"`, `mode = "capability"`. A judge model reads the request and estimates the probability that the weak model can handle it. At or above 0.5 the weak model answers; otherwise the strong one does.
  - `sy-random`: `type = "random"`, 50/50.
  - `sy-random-matched`: random, weighted 42:8 to match the classifier's observed split, so it spends the same budget with no judgement.
- **Models:** strong = `nvidia/nemotron-3-ultra-550b-a55b`, weak = `nvidia/nemotron-3.5-lightning-30b-a3b`, judge = `openai/gpt-oss-20b` at low reasoning effort. Each target gets `extra_body = { chat_template_kwargs = { enable_thinking = false } }`.
- **Which model served each request** comes from Switchyard's `x-model-router-selected-model` response header. Judge token usage comes from its `/v1/stats` endpoint.

Two configuration lessons:
- **The judge can't share a model with a candidate.** Switchyard keys execution by model ID within a route, so making the weak model its own judge is rejected at load time. A second client entry for the same endpoint doesn't get around it.
- **Disable thinking explicitly.** Nemotron 3.x reasons before answering by default, which adds seconds to every turn; `extra_body` on each target turns it off without changing callers.

## Method

`scripts/route_compare.py` sends the same 50 prompts (30 unique, cycled) through six strategies, with the same system prompt, `temperature 0.3` and `max_tokens 256`:

| Strategy | How the model is chosen |
|---|---|
| always-ultra | Every request to the strong model (quality ceiling) |
| always-lightning | Every request to the weak model (cost floor) |
| inferedge-semantic | InferEdge's semantic router (hashed embeddings, no LLM call) |
| sy-classifier | Switchyard LLM classifier |
| sy-random | Switchyard random, 50/50 |
| sy-random-matched | Switchyard random at the classifier's budget |

All 300 answers were then graded blind, 1 to 5, by `openai/gpt-oss-20b`. That model is not one of the two being routed between, and the grader never sees which model wrote the answer. The rubric rewards accurate, specific, actionable replies that escalate when asked and are honest that the assistant can't see the customer's account. Costs use InferEdge's assumed per-token prices (`MODEL_TIERS` in `backend/app/config.py`), with judge tokens priced at the small tier's rate.

## Results

Measured October 25, 2026 on the hosted NIM free tier.

| Strategy | Weak-model share | Quality (1-5) | Quality, hard requests | Good or better | TTFT p50 (ms) | TTFT p95 (ms) | Cost incl. judge | vs always-Ultra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always-ultra | 0% | 3.78 | 3.8 | 66% | 660 | 2,707 | $0.00739 | +0% |
| always-lightning | 100% | 3.66 | 3.8 | 70% | 233 | 308 | $0.00086 | −88% |
| inferedge-semantic | 74% | 3.68 | 4.2 | 70% | 247 | 1,636 | $0.00257 | −65% |
| sy-classifier | 16% | 3.92 | 4.3 | 74% | 3,874 | 7,577 | $0.01115 | +51% |
| sy-random | 48% | 3.88 | 3.7 | 72% | 356 | 2,079 | $0.00483 | −35% |
| sy-random-matched | 18% | 3.72 | 3.7 | 64% | 396 | 10,529 | $0.00649 | −12% |

"Hard requests" are the technical-support and escalation prompts (10 of the 50).

Quality relative to always-ultra, paired by prompt, with 95% bootstrap intervals:

| Strategy | Mean difference | 95% interval |
|---|---:|---|
| always-lightning | −0.12 | [−0.54, +0.30] |
| inferedge-semantic | −0.10 | [−0.44, +0.22] |
| sy-classifier | +0.14 | [−0.12, +0.40] |
| sy-random | +0.10 | [−0.26, +0.46] |
| sy-random-matched | −0.06 | [−0.34, +0.20] |
| sy-classifier vs sy-random-matched | +0.20 | [−0.08, +0.50] |

## What it shows

1. **Quality was a statistical tie.** With 50 prompts, none of the strategies differed from always-Ultra by more than the noise. Even always-Lightning, a model with about 3B active parameters, was indistinguishable from Ultra (about 55B active) on these short support turns. Routing decisions here should be made on cost and latency.
2. **On cost and latency the differences are large.** Always-Lightning was 88% cheaper than always-Ultra, with a 2.8x faster median first token. InferEdge's semantic router was 65% cheaper, added no measurable routing time, and kept the strong model for the hard requests (its 4.2 on hard requests vs 3.8 for always-Ultra is suggestive, not significant).
3. **The LLM classifier cost more than it saved: +51% vs always-Ultra.** Each verdict used about 1,125 tokens, mostly Switchyard's built-in judging instructions, while an average answer here is about 144 tokens. Judging a request cost more than answering it with the big model. The judge also added about 3.1 seconds before the first token (median first token 3.9 s vs 0.66 s), and it was conservative, sending 84% of requests to Ultra, including 9 of 10 routine billing prompts.
4. **The classifier did pick better than chance at the same budget, but not significantly.** Against budget-matched random routing, its quality was +0.20 higher, with an interval that just includes zero. A larger prompt set would settle it.
5. **Where LLM routing pays off is a matter of ratio.** An LLM judge makes sense when the judge is cheap relative to the work it routes: long agentic or coding tasks (Switchyard's own benchmarks are Terminal-Bench and SWE-style tasks), where one verdict decides on thousands of output tokens. For short, real-time support turns, a millisecond embedding router (or simply the small model) is the better trade. InferEdge's MiniLM router reached 100% model-tier accuracy on held-out messages ([README](../README.md#the-cuda-semantic-router)).

## Caveats

- **Small sample:** 50 prompts, 30 of them unique, and only 10 in the hard categories.
- **Single grader:** one model graded every answer, with no human check.
- **Assumed prices:** costs use assumed per-token rates; the free tier itself charges nothing.
- **Shared-tier latency:** the hosted free tier adds variable queueing, which is why a few tail latencies (for example sy-random-matched's p95) are much worse than their medians.
- **One configuration:** the classifier ran with Switchyard's default capability prompt and a 0.5 threshold. A custom prompt, a lower threshold, or `classify_trigger = "new_session"` (judge once per conversation instead of every turn) would change the cost picture.

## Reproduce

```bash
cargo install --locked switchyard-server --version 0.3.0
NVIDIA_API_KEY=nvapi-... switchyard-server --config switchyard/nim_routes.toml \
    --host 127.0.0.1 --port 4000 --routing-log-file benchmark_results/switchyard/routing_log.jsonl
backend/.venv/bin/python scripts/route_compare.py run \
    --strategies always-ultra,always-lightning,inferedge-semantic,sy-classifier,sy-random
backend/.venv/bin/python scripts/route_compare.py run --strategies sy-random-matched
backend/.venv/bin/python scripts/route_compare.py grade
backend/.venv/bin/python scripts/route_compare.py report
```

Raw results: [`benchmark_results/switchyard/`](../benchmark_results/switchyard/).

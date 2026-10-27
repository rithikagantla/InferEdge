"""InferEdge API — GPU-optimized LLM inference platform for real-time AI agents."""
import glob
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import benchmark as benchmark_runner
from . import config, db, service
from .metrics import tracker
from .routing import get_router
from .sample_prompts import SAMPLE_PROMPTS
from .schemas import (
    OPTIMIZATION_MODES,
    BatchChatRequest,
    BatchChatResponse,
    BenchmarkResult,
    BenchmarkRunRequest,
    ChatRequest,
    ChatResponse,
    RouteRequest,
    RouteResponse,
    RouterBenchmarkRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("inferedge")

@asynccontextmanager
async def lifespan(_: FastAPI):
    router = get_router()  # builds centroids, detects GPU
    logger.info(
        "InferEdge up — inference_mode=%s router_backend=%s",
        config.INFERENCE_MODE,
        router.backend,
    )
    yield


app = FastAPI(
    title="InferEdge",
    description="GPU-optimized LLM inference platform for real-time AI agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo platform; lock down for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "service": "InferEdge",
        "inference_mode": config.INFERENCE_MODE,
        "router_backend": get_router().backend,
        "optimization_modes": OPTIMIZATION_MODES,
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if req.mode not in OPTIMIZATION_MODES:
        raise HTTPException(422, "mode must be one of %s" % OPTIMIZATION_MODES)
    if req.mode in benchmark_runner.MOCK_ONLY_MODES and config.INFERENCE_MODE != "mock":
        raise HTTPException(
            422, "mode %r is simulated and only available with INFERENCE_MODE=mock" % req.mode
        )
    try:
        return await service.chat(req.prompt, mode=req.mode,
                                  history=[m.model_dump() for m in req.history])
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc))
    except Exception as exc:  # provider/network errors surface as 502
        logger.exception("chat failed")
        raise HTTPException(502, "inference failed: %s" % exc)


@app.post("/batch-chat", response_model=BatchChatResponse)
async def batch_chat(req: BatchChatRequest) -> BatchChatResponse:
    try:
        return await service.batch_chat(req.prompts)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc))
    except Exception as exc:
        logger.exception("batch chat failed")
        raise HTTPException(502, "inference failed: %s" % exc)


@app.get("/metrics")
async def metrics() -> dict:
    snap = tracker.snapshot()
    snap["router_benchmark"] = get_router().last_benchmark
    snap["cache"] = {
        "hits": service.prompt_cache.hits,
        "misses": service.prompt_cache.misses,
        "hit_rate": round(service.prompt_cache.hit_rate, 3),
    }
    return snap


@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest) -> RouteResponse:
    return RouteResponse(**get_router().classify(req.prompt))


@app.post("/router/benchmark")
async def router_benchmark(req: RouterBenchmarkRequest) -> dict:
    return get_router().benchmark(req.num_intents)


@app.post("/benchmark/run", response_model=BenchmarkResult)
async def run_benchmark(req: BenchmarkRunRequest) -> BenchmarkResult:
    try:
        return await benchmark_runner.run_benchmark(req.num_prompts, req.modes)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc))


@app.get("/benchmark-results")
async def benchmark_results(limit: int = 20) -> dict:
    runs = db.load_benchmarks(limit=min(limit, 100))
    return {"runs": [r.model_dump() for r in runs]}


@app.get("/sample-prompts")
async def sample_prompts() -> dict:
    return {"prompts": SAMPLE_PROMPTS}


MODE_DESCRIPTIONS = {
    "routing": "Semantic router picks the small or large model per message",
    "baseline": "Every message goes to the large model",
    "caching": "Repeated single-turn questions are answered from an LRU cache",
    "quantized": "Simulated INT8/FP8 large model (mock mode only)",
}


@app.get("/config")
async def get_config() -> dict:
    """Models, prices and modes, so the dashboard can explain each number."""
    router = get_router()
    from .routing.semantic_router import SMALL_MODEL_CATEGORIES

    modes = [m for m in ("routing", "baseline", "caching", "quantized")
             if m not in benchmark_runner.MOCK_ONLY_MODES or config.INFERENCE_MODE == "mock"]
    return {
        "inference_mode": config.INFERENCE_MODE,
        "router_backend": router.backend,
        "router_embeddings": router.embeddings,
        "small_tier_categories": sorted(SMALL_MODEL_CATEGORIES),
        "tiers": {
            name: {"model": t.name, "input_cost_per_1m": t.input_cost_per_1m,
                   "output_cost_per_1m": t.output_cost_per_1m}
            for name, t in config.MODEL_TIERS.items()
        },
        "chat_modes": [{"id": m, "description": MODE_DESCRIPTIONS[m]} for m in modes],
    }


RESULTS_DIR = os.path.join(config._REPO_ROOT, "benchmark_results")


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@app.get("/results")
async def measured_results() -> dict:
    """Measured (not simulated) results committed under benchmark_results/:
    hosted NIM runs, self-hosted GPU sweeps, router GPU benchmarks and the
    Switchyard routing study. Missing files are simply omitted."""
    sweep_keys = ("concurrency", "requests", "errors", "output_tokens_per_sec",
                  "requests_per_sec", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms",
                  "tpot_p95_ms", "itl_p99_ms", "e2e_p50_ms", "avg_output_tokens")
    meta_keys = ("label", "engine", "engine_version", "precision", "variant", "trtllm_backend",
                 "gpu_name", "gpu_memory_peak_used_mib", "ignore_eos", "model", "max_tokens",
                 "speculative_config", "max_batch_size", "max_num_tokens", "created_at")

    def sweeps(pattern: str) -> list:
        out = []
        for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
            d = _read_json(path)
            if not d or not d.get("levels"):
                continue
            out.append({**{k: d.get(k) for k in meta_keys},
                        "levels": [{k: lv.get(k) for k in sweep_keys} for lv in d["levels"]]})
        return out

    nim_runs = sorted(glob.glob(os.path.join(RESULTS_DIR, "nim", "benchmark_nim_*.json")))
    nim_benchmark = (_read_json(nim_runs[-1]) or {}).get("benchmark") if nim_runs else None
    router_embed = _read_json(os.path.join(RESULTS_DIR, "gpu", "router_embed_L4.json"))
    if router_embed:
        router_embed = {k: v for k, v in router_embed.items() if k != "accuracy"} | {
            "accuracy": {name: {k: v for k, v in a.items() if k != "misses"}
                         for name, a in router_embed["accuracy"].items()}}
    return {
        "nim_benchmark": nim_benchmark,
        "nim_sweeps": sweeps("nim/sweep_*.json"),
        "gpu_sweeps": sweeps("gpu/sweep_*-L4.json"),
        "router_gpu": _read_json(os.path.join(RESULTS_DIR, "gpu", "router_L4.json")),
        "router_embed": router_embed,
        "routing_study": _read_json(os.path.join(RESULTS_DIR, "switchyard", "route_compare.json")),
    }

"""InferEdge API — GPU-optimized LLM inference platform for real-time AI agents."""
import logging
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
        return await service.chat(req.prompt, mode=req.mode)
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

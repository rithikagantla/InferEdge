"""CUDA-accelerated semantic router with CPU fallback.

Classifies a support prompt into one of six intent categories and maps
it to a model tier (small/large), so cheap intents don't burn 70B-class
compute.

How it works
------------
1. Embeddings: each token is hashed to a seed that generates a fixed
   random Gaussian vector (a random-projection / feature-hashing
   embedding). Token vectors are summed and L2-normalized. This is a
   deterministic, dependency-free stand-in for a real sentence encoder —
   swap `embed()` for e.g. NVIDIA NeMo Retriever / sentence-transformers
   without touching anything else.
2. Each category has a centroid: the normalized mean embedding of a set
   of exemplar phrases.
3. Classification = cosine similarity of the prompt embedding against
   the centroid matrix — one (num_categories x dim) @ (dim,) matvec.

Where CUDA comes in
-------------------
Similarity scoring is dense linear algebra, so it runs on:
  * GPU via CuPy — the matmul executes as a cuBLAS kernel on device.
    Arrays are moved to GPU memory once at startup (centroids) and per
    call for the query. `cp.cuda.Stream.null.synchronize()` is required
    before reading timings because CUDA kernel launches are async.
  * CPU via NumPy — automatic fallback when CuPy or an NVIDIA GPU is
    unavailable ("CPU fallback mode").

A single query matvec is too small to show off a GPU, which is why
`benchmark()` scores a large batch of intents at once (N x dim matrix)
on both backends and reports the speedup — that mirrors scoring a
concurrent stream of production traffic.
"""
import time
import zlib
from typing import Dict, List, Optional

import numpy as np

# ---- GPU detection -------------------------------------------------------
# CuPy is an optional dependency: it requires an NVIDIA GPU + CUDA toolkit.
# Everything below degrades gracefully to NumPy when it is missing.
try:  # pragma: no cover - exercised only on GPU machines
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()  # raises if no NVIDIA GPU/driver
    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False

EMBED_DIM = 384

CATEGORY_EXEMPLARS: Dict[str, List[str]] = {
    "billing": [
        "I was charged twice this month",
        "why is my bill higher than usual",
        "I need a copy of my invoice",
        "my payment failed but I was charged",
        "update my credit card billing information",
        "question about a charge on my statement",
    ],
    "technical_support": [
        "the app keeps crashing after the update",
        "I get an error when I try to check out",
        "the website will not load on my browser",
        "my data is not syncing between devices",
        "the api times out on large uploads",
        "video playback is laggy and buffering",
    ],
    "account_management": [
        "I forgot my password and cannot log in",
        "reset my password I am locked out of my account",
        "can I upgrade my subscription plan",
        "change the email address on my account",
        "add another user to my team plan",
        "enable two factor authentication",
        "downgrade or cancel my subscription plan",
    ],
    "refund": [
        "I want a refund for my last order",
        "how long does a refund take to process",
        "I returned the item but got no money back",
        "requesting a partial refund for damaged goods",
        "what is your refund policy",
        "give me my money back",
    ],
    "shipping": [
        "my order is delayed can you help",
        "where is my package right now",
        "tracking number has not updated in days",
        "change the delivery address for my order",
        "do you ship internationally",
        "my package arrived damaged in the mail",
    ],
    "human_escalation": [
        "let me talk to a real person",
        "I need to speak with a human agent",
        "connect me to a support representative",
        "I want to file a complaint with a manager",
        "escalate this ticket immediately",
        "your bot is not helping me at all",
    ],
}

# Intents the small (8B-class) model handles well; everything else goes
# to the large model. human_escalation also goes large so the handoff
# summary is high quality.
SMALL_MODEL_CATEGORIES = {"billing", "shipping", "refund", "account_management"}


def _token_seed(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


def embed(text: str) -> np.ndarray:
    """Deterministic hashed random-projection embedding (CPU).

    Embedding is cheap; the hot path we accelerate is similarity scoring.
    """
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        return vec
    for token in tokens:
        rng = np.random.default_rng(_token_seed(token))
        vec += rng.standard_normal(EMBED_DIM, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class SemanticRouter:
    def __init__(self) -> None:
        self.categories = list(CATEGORY_EXEMPLARS.keys())
        centroids = []
        for category in self.categories:
            exemplar_matrix = np.stack([embed(p) for p in CATEGORY_EXEMPLARS[category]])
            centroid = exemplar_matrix.mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            centroids.append(centroid)
        # (num_categories, EMBED_DIM) on CPU...
        self.centroids_cpu = np.stack(centroids).astype(np.float32)
        # ...and resident in GPU memory once, if we have one.
        self.centroids_gpu = cp.asarray(self.centroids_cpu) if GPU_AVAILABLE else None
        self.backend = "cuda (CuPy)" if GPU_AVAILABLE else "cpu (NumPy fallback)"
        self.last_benchmark: Optional[dict] = None

    # -- classification ----------------------------------------------------
    def classify(self, prompt: str) -> dict:
        query = embed(prompt)
        start = time.perf_counter()
        if GPU_AVAILABLE:
            # Device matvec (cuBLAS); synchronize so the timing is honest.
            scores_dev = self.centroids_gpu @ cp.asarray(query)
            cp.cuda.Stream.null.synchronize()
            scores = cp.asnumpy(scores_dev)
        else:
            scores = self.centroids_cpu @ query
        route_time_ms = (time.perf_counter() - start) * 1000

        best = int(np.argmax(scores))
        category = self.categories[best]
        # Softmax over similarities → a friendlier confidence number.
        exp = np.exp((scores - scores.max()) * 8.0)
        confidence = float(exp[best] / exp.sum())
        return {
            "category": category,
            "confidence": round(confidence, 4),
            "scores": {c: round(float(s), 4) for c, s in zip(self.categories, scores)},
            "model_tier": "small" if category in SMALL_MODEL_CATEGORIES else "large",
            "backend": self.backend,
            "route_time_ms": round(route_time_ms, 4),
        }

    # -- CPU vs GPU benchmark ------------------------------------------------
    def benchmark(self, num_intents: int = 2000) -> dict:
        """Score `num_intents` simulated intent embeddings against the
        centroids on CPU and (if available) GPU, and report the speedup.
        """
        rng = np.random.default_rng(42)
        queries = rng.standard_normal((num_intents, EMBED_DIM), dtype=np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        # CPU (NumPy / BLAS). errstate silences spurious FP-status
        # warnings from Apple's Accelerate BLAS on macOS.
        with np.errstate(all="ignore"):
            start = time.perf_counter()
            cpu_scores = queries @ self.centroids_cpu.T
            cpu_scores.argmax(axis=1)
            cpu_ms = (time.perf_counter() - start) * 1000

        gpu_ms = None
        speedup = None
        if GPU_AVAILABLE:
            queries_gpu = cp.asarray(queries)
            # Warm-up launch so we don't time one-time CUDA context /
            # kernel compilation cost.
            (queries_gpu @ self.centroids_gpu.T).argmax(axis=1)
            cp.cuda.Stream.null.synchronize()

            start = time.perf_counter()
            gpu_scores = queries_gpu @ self.centroids_gpu.T
            gpu_scores.argmax(axis=1)
            cp.cuda.Stream.null.synchronize()  # kernels are async — wait
            gpu_ms = (time.perf_counter() - start) * 1000
            speedup = cpu_ms / gpu_ms if gpu_ms > 0 else None

        self.last_benchmark = {
            "num_intents": num_intents,
            "embed_dim": EMBED_DIM,
            "num_categories": len(self.categories),
            "cpu_time_ms": round(cpu_ms, 3),
            "gpu_time_ms": round(gpu_ms, 3) if gpu_ms is not None else None,
            "speedup": round(speedup, 2) if speedup is not None else None,
            "gpu_available": GPU_AVAILABLE,
            "backend": self.backend,
            "note": (
                "GPU (CuPy/cuBLAS) vs CPU (NumPy) similarity scoring"
                if GPU_AVAILABLE
                else "CPU fallback mode — install CuPy on an NVIDIA GPU machine to enable CUDA scoring"
            ),
        }
        return self.last_benchmark


_router: Optional[SemanticRouter] = None


def get_router() -> SemanticRouter:
    global _router
    if _router is None:
        _router = SemanticRouter()
    return _router

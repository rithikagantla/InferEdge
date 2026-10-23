"""CPU vs GPU benchmark of the InferEdge semantic router on one NVIDIA L4 (Modal).

    modal run gpu/modal_router.py        # ~2 minutes of L4 time

The router (backend/app/routing/semantic_router.py) scores intent
embeddings against category centroids with one matrix multiply. This runs
that exact code on the container's CPU (NumPy) and on the L4 (CuPy /
cuBLAS) across batch sizes from 1 to 1,000,000 intents, and reports three
timings per size:

  cpu_ms              NumPy matmul + argmax
  gpu_compute_ms      CuPy matmul + argmax, data already on the GPU
  gpu_end_to_end_ms   copy the batch to the GPU, score it, copy the winners
                      back: what a request handler actually pays

Each is the median of several runs after a warm-up, and GPU timings
synchronize the CUDA stream before stopping the clock, since kernel
launches are asynchronous. Results land in benchmark_results/gpu/.
"""
import modal

from common import GPU, REPO_ROOT, add_repo_code

MINUTES = 60
SIZES = [1, 100, 1_000, 10_000, 100_000, 1_000_000]
REPEATS = 7

image = add_repo_code(
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("numpy<2.3", "cupy-cuda12x==13.6.0", "httpx")
)

app = modal.App("inferedge-router", image=image)


@app.function(gpu=GPU, timeout=15 * MINUTES)
def bench() -> dict:
    import platform
    import statistics
    import subprocess
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/root/inferedge/backend")
    from app.routing import semantic_router as sr

    if not sr.GPU_AVAILABLE:
        raise RuntimeError("CuPy could not see the GPU; refusing to report CPU-only numbers")
    cp = sr.cp
    router = sr.SemanticRouter()

    def median_ms(fn) -> float:
        fn()  # warm-up: CUDA context, kernel selection, caches
        runs = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn()
            runs.append((time.perf_counter() - t0) * 1000)
        return statistics.median(runs)

    rows = []
    rng = np.random.default_rng(42)
    for n in SIZES:
        q = rng.standard_normal((n, sr.EMBED_DIM), dtype=np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        q_gpu = cp.asarray(q)
        cp.cuda.Stream.null.synchronize()

        def cpu():
            (q @ router.centroids_cpu.T).argmax(axis=1)

        def gpu_compute():
            (q_gpu @ router.centroids_gpu.T).argmax(axis=1)
            cp.cuda.Stream.null.synchronize()

        def gpu_end_to_end():
            cp.asnumpy((cp.asarray(q) @ router.centroids_gpu.T).argmax(axis=1))

        with np.errstate(all="ignore"):
            cpu_ms = median_ms(cpu)
        gpu_ms = median_ms(gpu_compute)
        e2e_ms = median_ms(gpu_end_to_end)
        rows.append({
            "intents": n,
            "cpu_ms": round(cpu_ms, 4),
            "gpu_compute_ms": round(gpu_ms, 4),
            "gpu_end_to_end_ms": round(e2e_ms, 4),
            "speedup_compute": round(cpu_ms / gpu_ms, 2),
            "speedup_end_to_end": round(cpu_ms / e2e_ms, 2),
        })
        print(rows[-1], flush=True)
        del q_gpu

    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    return {
        "gpu": gpu_name,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "embed_dim": sr.EMBED_DIM,
        "categories": len(router.categories),
        "repeats": REPEATS,
        "rows": rows,
    }


def markdown(result: dict) -> str:
    lines = [
        "Semantic router similarity scoring on %s vs the container CPU (%d cores), "
        "%d-dim embeddings x %d categories, median of %d runs.\n"
        % (result["gpu"], result["cpu_count"], result["embed_dim"], result["categories"],
           result["repeats"]),
        "| Intents | CPU NumPy (ms) | GPU compute (ms) | GPU incl. transfer (ms) "
        "| Speedup (compute) | Speedup (incl. transfer) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["rows"]:
        lines.append(
            "| {intents:,} | {cpu_ms:.3f} | {gpu_compute_ms:.3f} | {gpu_end_to_end_ms:.3f} "
            "| {speedup_compute:.2f}x | {speedup_end_to_end:.2f}x |".format(**r)
        )
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main():
    import json

    result = bench.remote()
    out = REPO_ROOT / "benchmark_results" / "gpu"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("router_%s.json" % GPU)).write_text(json.dumps(result, indent=2))
    (out / ("router_%s.md" % GPU)).write_text(markdown(result))
    print("\n" + markdown(result))
    print("saved", out / ("router_%s.json" % GPU), "and .md")

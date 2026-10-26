"""Semantic router with real embeddings on an NVIDIA L4 (Modal): accuracy and latency.

    modal run gpu/modal_router_embed.py      # a few minutes of L4 time

1. Accuracy on the held-out labeled set (backend/app/router_eval_set.py):
   hashed word embeddings (the default router) vs all-MiniLM-L6-v2.
2. End-to-end routing latency for batches of 1 to 10,000 messages, text in,
   category out (tokenization + embedding + scoring + argmax):
     cpu            MiniLM and scoring on the container's CPU cores
     gpu_resident   MiniLM, scoring and argmax on the GPU; only the winning
                    category indices are copied back
     gpu_embed_only MiniLM on the GPU, embeddings copied back, scored on CPU

Results land in benchmark_results/gpu/router_embed_<GPU>.{json,md}.
"""
import modal

from common import GPU, REPO_ROOT, add_repo_code

MINUTES = 60
SIZES = [1, 10, 100, 1_000, 10_000]
CPU_CORES = 8.0

image = add_repo_code(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.7.0", "sentence-transformers==5.1.1", "numpy<2.3", "httpx")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("inferedge-router-embed", image=image)


@app.function(gpu=GPU, cpu=CPU_CORES, timeout=30 * MINUTES,
              volumes={"/root/.cache/huggingface": hf_cache})
def bench() -> dict:
    import statistics
    import subprocess
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root/inferedge/backend")
    from app.router_eval_set import ROUTER_EVAL_SET
    from app.routing import semantic_router as sr

    texts = [p for p, _ in ROUTER_EVAL_SET]
    labels = [c for _, c in ROUTER_EVAL_SET]

    def tier(c):
        return "small" if c in sr.SMALL_MODEL_CATEGORIES else "large"

    def accuracy(pred):
        return {
            "category_accuracy": round(sum(p == y for p, y in zip(pred, labels)) / len(labels), 3),
            "tier_accuracy": round(sum(tier(p) == tier(y) for p, y in zip(pred, labels)) / len(labels), 3),
            "misses": [(t, y, p) for t, y, p in zip(texts, labels, pred) if p != y],
        }

    hashed = sr.SemanticRouter("hashed")
    minilm_gpu = sr.SemanticRouter("minilm")
    assert minilm_gpu.embedder.device == "cuda", "MiniLM did not land on the GPU"
    cpu_embedder = sr.MiniLMEmbedder(device="cpu")
    centroids_cpu_t = minilm_gpu.centroids_t.cpu()

    acc = {"hashed": accuracy(hashed.classify_batch(texts)),
           "minilm": accuracy(minilm_gpu.classify_batch(texts))}
    for name, a in acc.items():
        print(name, "category %.0f%%  tier %.0f%%" % (100 * a["category_accuracy"], 100 * a["tier_accuracy"]),
              flush=True)

    def cpu_route(batch):
        scores = cpu_embedder.encode(batch) @ centroids_cpu_t.T
        return scores.argmax(dim=1).tolist()

    def gpu_resident(batch):
        idx = (minilm_gpu.embedder.encode(batch) @ minilm_gpu.centroids_t.T).argmax(dim=1)
        return idx.cpu().tolist()  # .cpu() synchronizes the stream

    def gpu_embed_only(batch):
        emb = minilm_gpu.embedder.encode(batch).cpu().numpy()
        return (emb @ minilm_gpu.centroids_cpu.T).argmax(axis=1).tolist()

    def median_ms(fn, batch, repeats):
        fn(batch)  # warm-up
        runs = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(batch)
            runs.append((time.perf_counter() - t0) * 1000)
        return statistics.median(runs)

    rows = []
    for n in SIZES:
        batch = [texts[i % len(texts)] for i in range(n)]
        # Same answers on every path before any timing counts.
        ref = gpu_resident(batch)
        assert cpu_route(batch) == ref, "CPU and GPU routes disagree"
        assert gpu_embed_only(batch) == ref, "GPU-embed/CPU-score disagrees"
        repeats = 7 if n <= 1000 else 3
        row = {"messages": n,
               "cpu_ms": median_ms(cpu_route, batch, repeats),
               "gpu_resident_ms": median_ms(gpu_resident, batch, repeats),
               "gpu_embed_only_ms": median_ms(gpu_embed_only, batch, repeats)}
        row["speedup_gpu_resident"] = round(row["cpu_ms"] / row["gpu_resident_ms"], 2)
        row["gpu_messages_per_s"] = round(n / (row["gpu_resident_ms"] / 1000))
        row["cpu_messages_per_s"] = round(n / (row["cpu_ms"] / 1000))
        for k in ("cpu_ms", "gpu_resident_ms", "gpu_embed_only_ms"):
            row[k] = round(row[k], 3)
        rows.append(row)
        print(row, flush=True)

    # Hashed router on CPU for reference (its per-message cost is Python hashing).
    hashed_rows = []
    for n in SIZES:
        batch = [texts[i % len(texts)] for i in range(n)]
        hashed_rows.append({"messages": n, "cpu_ms": round(median_ms(hashed.classify_batch, batch, 3), 3)})

    hf_cache.commit()
    return {
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       text=True).strip(),
        "cpu_cores": CPU_CORES,
        "torch": str(torch.__version__),
        "model": sr.MINILM_MODEL,
        "eval_set_size": len(texts),
        "accuracy": acc,
        "latency": rows,
        "hashed_cpu_latency": hashed_rows,
    }


def markdown(r: dict) -> str:
    a = r["accuracy"]
    lines = [
        "Held-out accuracy (%d labeled support messages):\n" % r["eval_set_size"],
        "| Embeddings | Category accuracy | Model-tier accuracy |",
        "|---|---:|---:|",
        "| Hashed words (default) | %.0f%% | %.0f%% |" % (100 * a["hashed"]["category_accuracy"],
                                                     100 * a["hashed"]["tier_accuracy"]),
        "| all-MiniLM-L6-v2 | %.0f%% | %.0f%% |" % (100 * a["minilm"]["category_accuracy"],
                                                 100 * a["minilm"]["tier_accuracy"]),
        "",
        "End-to-end routing latency, text in to category out, on %s vs %d CPU cores "
        "(median of several runs):\n" % (r["gpu"], r["cpu_cores"]),
        "| Messages | CPU (ms) | GPU, all on device (ms) | GPU embed, CPU score (ms) | Speedup | GPU messages/s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for x in r["latency"]:
        lines.append("| {messages:,} | {cpu_ms:.1f} | {gpu_resident_ms:.1f} | {gpu_embed_only_ms:.1f} "
                     "| {speedup_gpu_resident:.1f}x | {gpu_messages_per_s:,} |".format(**x))
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main():
    import json

    result = bench.remote()
    out = REPO_ROOT / "benchmark_results" / "gpu"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("router_embed_%s.json" % GPU)).write_text(json.dumps(result, indent=2))
    (out / ("router_embed_%s.md" % GPU)).write_text(markdown(result))
    print("\n" + markdown(result))

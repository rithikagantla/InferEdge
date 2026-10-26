"""Self-hosted vLLM on one NVIDIA L4 (Modal), measured with the InferEdge sweep.

    modal run gpu/modal_vllm.py --smoke                 # ~5 min check, bf16, c=1,2
    modal run gpu/modal_vllm.py --variants bf16,fp8     # full sweeps
    modal run gpu/modal_vllm.py --variants fp8-kv       # FP8 weights + FP8 KV cache
    modal serve gpu/modal_vllm.py                       # temporary public endpoint
                                                        # for INFERENCE_MODE=local

The sweep runs inside the GPU container against localhost, so internet
round trips don't pollute TTFT or inter-token latency. Results land in
benchmark_results/gpu/ on your machine.

bf16 is the checkpoint's native precision. fp8 is vLLM's dynamic FP8
weight quantization (no calibration), which the L4's Ada tensor cores run
natively: weights shrink from ~16 GB to ~8.5 GB, leaving more memory for
KV cache and therefore more concurrent sequences.
"""
import time
from importlib.metadata import version

import modal

from common import GPU, MODEL_ID, MODEL_REVISION, PORT, add_repo_code, save_outputs

MINUTES = 60
VLLM_VERSION = "0.21.0"  # pinned to the version in Modal's maintained vLLM example
FULL_LEVELS = [1, 2, 4, 8, 16, 32, 64]
SMOKE_LEVELS = [1, 2]

image = add_repo_code(
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==" + VLLM_VERSION, "httpx")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache}

app = modal.App("inferedge-vllm", image=image)


def vllm_cmd(precision: str, host: str = "127.0.0.1") -> list:
    cmd = [
        "vllm", "serve", MODEL_ID, "--revision", MODEL_REVISION,
        "--host", host, "--port", str(PORT),
        # 2048 covers short support prompts + 256 output tokens; a shorter
        # max length means less KV cache reserved per sequence.
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "128",
    ]
    if precision == "fp8":
        cmd += ["--quantization", "fp8"]
    elif precision == "fp8-kv":
        # FP8 weights *and* KV cache, the like-for-like comparison with the
        # TensorRT-LLM FP8 engine (which quantizes its KV cache too).
        cmd += ["--quantization", "fp8", "--kv-cache-dtype", "fp8"]
    elif precision != "bf16":
        raise ValueError("precision must be bf16, fp8 or fp8-kv")
    return cmd


@app.function(gpu=GPU, volumes=VOLUMES, timeout=45 * MINUTES)
def sweep(precision: str = "bf16", smoke: bool = False) -> dict:
    from common import GpuSampler, Server, run_sweep_and_package

    label = "vllm-%s-%s%s" % (precision, GPU, "-smoke" if smoke else "")
    cmd = vllm_cmd(precision)
    with GpuSampler() as sampler:
        server = Server(cmd)
        try:
            t0 = time.time()
            server.wait_ready()
            meta = {
                "engine": "vLLM",
                "engine_version": version("vllm"),
                "precision": precision,
                "hf_model": MODEL_ID,
                "hf_revision": MODEL_REVISION,
                "server_startup_s": round(time.time() - t0, 1),
                "server_cmd": " ".join(cmd),
            }
            return run_sweep_and_package(
                label, SMOKE_LEVELS if smoke else FULL_LEVELS, meta, server, sampler
            )
        finally:
            server.stop()
            hf_cache.commit()
            vllm_cache.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=60 * MINUTES, scaledown_window=5 * MINUTES)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=20 * MINUTES)
def serve():
    """`modal serve gpu/modal_vllm.py` exposes this as a URL; the GPU shuts
    down 5 minutes after the last request (and when you Ctrl+C)."""
    import subprocess

    subprocess.Popen(vllm_cmd("bf16", host="0.0.0.0"))


@app.local_entrypoint()
def main(variants: str = "bf16,fp8", smoke: bool = False):
    if smoke:
        variants = variants.split(",")[0]
    for precision in [v.strip() for v in variants.split(",") if v.strip()]:
        print("=== vLLM %s on %s%s ===" % (precision, GPU, " (smoke)" if smoke else ""))
        save_outputs(sweep.remote(precision, smoke))

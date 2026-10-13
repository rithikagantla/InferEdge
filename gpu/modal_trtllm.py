"""TensorRT-LLM on one NVIDIA L4 (Modal): build engines, serve, sweep.

    modal run gpu/modal_trtllm.py --smoke                # bf16 engine, c=1,2
    modal run gpu/modal_trtllm.py --variants bf16,fp8    # full sweeps

Unlike vLLM, TensorRT-LLM compiles the model ahead of time into a
TensorRT engine specialized for this GPU (sm_89 / Ada), this batch size
and these sequence lengths. Engines are cached in the `inferedge-trtllm`
Volume, so the slow build happens once per precision.

fp8 uses post-training quantization: TensorRT-LLM (via NVIDIA ModelOpt)
runs calibration batches through the bf16 model to pick FP8 scaling
factors, then builds FP8 GEMM kernels. That calibration step is the main
difference from vLLM's calibration-free dynamic FP8.

Adapted from Modal's TensorRT-LLM example (tensorrt-llm 0.20.0, which
targets H100): retargeted to L4 (TORCH_CUDA_ARCH_LIST=8.9), the
Hopper-only low-latency FP8 GEMM plugins dropped, and calibration
shortened. See docs/TENSORRT_LLM.md for what worked and what didn't.
"""
import json
import os
import subprocess
import sys
import time

import modal

from common import GPU, MODEL_ID, MODEL_REVISION, PORT, add_repo_code, save_outputs

MINUTES = 60
TRTLLM_VERSION = "0.20.0"
FULL_LEVELS = [1, 2, 4, 8, 16, 32, 64]
SMOKE_LEVELS = [1, 2]
MAX_BATCH_SIZE = 64

VOL_PATH = "/vol"
MODEL_PATH = VOL_PATH + "/models/llama-3-8b-instruct"

image = add_repo_code(
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget")
    .uv_pip_install(
        "tensorrt-llm==" + TRTLLM_VERSION,
        "pynvml>=12",
        "flashinfer-python==0.2.5",
        "cuda-python==12.9.1",
        "onnx==1.19.1",
        "mpmath==1.3.0",
        "torch==2.7.0",
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .uv_pip_install("huggingface_hub==0.36.0", "httpx")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": VOL_PATH + "/hf",
        "TORCH_CUDA_ARCH_LIST": "8.9",  # L4 (Ada); Modal's H100 recipe uses 9.0
    })
)

volume = modal.Volume.from_name("inferedge-trtllm", create_if_missing=True)
app = modal.App("inferedge-trtllm", image=image)

# Runs in a fresh Python process so every byte of GPU memory used while
# building is released before trtllm-serve starts.
BUILD_SCRIPT = r'''
import json, sys, time
from tensorrt_llm import LLM, BuildConfig
from tensorrt_llm.llmapi import CalibConfig, QuantConfig
from tensorrt_llm.plugin.plugin import PluginConfig

precision, model_path, engine_dir, max_batch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
build_config = BuildConfig(
    plugin_config=PluginConfig.from_dict({"multiple_profiles": True, "paged_kv_cache": True}),
    max_input_len=2048,
    max_seq_len=2048,
    max_num_tokens=8192,
    max_batch_size=max_batch,
)
kwargs = {}
if precision == "fp8":
    kwargs["quant_config"] = QuantConfig(quant_algo="FP8", kv_cache_quant_algo="FP8")
    kwargs["calib_config"] = CalibConfig(
        calib_batches=64, calib_batch_size=1, calib_max_seq_length=1024,
        tokenizer_max_seq_length=2048,
    )
t0 = time.time()
llm = LLM(model=model_path, build_config=build_config, tensor_parallel_size=1, **kwargs)
llm.save(engine_dir)
build_s = time.time() - t0
llm.shutdown()
with open(engine_dir + "/inferedge_build.json", "w") as f:
    json.dump({"precision": precision, "build_s": round(build_s, 1), "max_batch_size": max_batch}, f)
print("ENGINE BUILT in %.0fs" % build_s, flush=True)
'''


def _download_model() -> None:
    from huggingface_hub import snapshot_download

    if os.path.exists(MODEL_PATH + "/config.json"):
        return
    snapshot_download(MODEL_ID, local_dir=MODEL_PATH, revision=MODEL_REVISION,
                      ignore_patterns=["*.pt", "*.bin", "original/*"])
    volume.commit()


def _ensure_engine(precision: str) -> dict:
    engine_dir = "%s/engines/%s-bs%d" % (VOL_PATH, precision, MAX_BATCH_SIZE)
    info_path = engine_dir + "/inferedge_build.json"
    if not os.path.exists(info_path):
        print("building %s engine (first run only)..." % precision, flush=True)
        subprocess.run(
            [sys.executable, "-c", BUILD_SCRIPT, precision, MODEL_PATH, engine_dir,
             str(MAX_BATCH_SIZE)],
            check=True,
        )
        volume.commit()
        cached = False
    else:
        cached = True
    with open(info_path) as f:
        info = json.load(f)
    info.update({"engine_dir": engine_dir, "engine_cached": cached})
    return info


@app.function(gpu=GPU, volumes={VOL_PATH: volume}, timeout=90 * MINUTES)
def sweep(precision: str = "bf16", smoke: bool = False) -> dict:
    from importlib.metadata import version

    from common import GpuSampler, Server, run_sweep_and_package

    if precision not in ("bf16", "fp8"):
        raise ValueError("precision must be bf16 or fp8")
    _download_model()
    build = _ensure_engine(precision)

    label = "trtllm-%s-%s%s" % (precision, GPU, "-smoke" if smoke else "")
    cmd = ["trtllm-serve", build["engine_dir"], "--tokenizer", MODEL_PATH,
           "--host", "127.0.0.1", "--port", str(PORT)]
    with GpuSampler() as sampler:
        server = Server(cmd)
        try:
            t0 = time.time()
            server.wait_ready()
            meta = {
                "engine": "TensorRT-LLM",
                "engine_version": version("tensorrt-llm"),
                "precision": precision,
                "hf_model": MODEL_ID,
                "hf_revision": MODEL_REVISION,
                "engine_build_s": build["build_s"],
                "engine_cached": build["engine_cached"],
                "max_batch_size": build["max_batch_size"],
                "server_startup_s": round(time.time() - t0, 1),
                "server_cmd": " ".join(cmd),
            }
            return run_sweep_and_package(
                label, SMOKE_LEVELS if smoke else FULL_LEVELS, meta, server, sampler
            )
        finally:
            server.stop()


@app.local_entrypoint()
def main(variants: str = "bf16,fp8", smoke: bool = False):
    if smoke:
        variants = variants.split(",")[0]
    for precision in [v.strip() for v in variants.split(",") if v.strip()]:
        print("=== TensorRT-LLM %s on %s%s ===" % (precision, GPU, " (smoke)" if smoke else ""))
        try:
            save_outputs(sweep.remote(precision, smoke))
        except Exception as exc:  # keep going: a failed fp8 build shouldn't lose bf16
            print("TensorRT-LLM %s FAILED: %s" % (precision, exc))

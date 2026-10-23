"""TensorRT-LLM on one NVIDIA L4 (Modal): build engines, serve, sweep.

    modal run gpu/modal_trtllm.py --smoke                          # bf16 engine, c=1,2
    modal run gpu/modal_trtllm.py --variants bf16,fp8,pytorch-bf16  # full sweeps

Three variants, all served with `trtllm-serve` and measured with the same
sweep as vLLM:

  bf16          Classic TensorRT-LLM: the model is compiled ahead of time
                into a TensorRT engine specialized for this GPU (sm_89 / Ada),
                batch size and sequence lengths, then served with
                `--backend tensorrt`.
  fp8           Same engine path with post-training FP8 quantization:
                NVIDIA ModelOpt runs calibration batches through the bf16
                model to pick FP8 scaling factors for weights, activations
                and the KV cache, then FP8 kernels are built. That
                calibration step is the main difference from vLLM's
                calibration-free dynamic FP8.
  pytorch-bf16  TensorRT-LLM 1.x's default backend: no engine build; the
                Hugging Face checkpoint is served directly with TensorRT-LLM's
                PyTorch runtime and kernels.

Engines are cached in the `inferedge-trtllm` Volume, so each build happens
once. Runs on NVIDIA's prebuilt TensorRT-LLM 1.2.1 container from NGC:
pip-installing tensorrt-llm 0.20.0 (Modal's original recipe) no longer
works because its pinned dependency xgrammar==0.1.16 was removed from
PyPI. See docs/TENSORRT_LLM.md.
"""
import json
import os
import subprocess
import sys
import time

import modal

from common import GPU, MODEL_ID, MODEL_REVISION, PORT, add_repo_code, save_outputs

MINUTES = 60
TRTLLM_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
FULL_LEVELS = [1, 2, 4, 8, 16, 32, 64]
SMOKE_LEVELS = [1, 2]
MAX_BATCH_SIZE = 64
MAX_SEQ_LEN = 2048
VARIANTS = ("bf16", "fp8", "pytorch-bf16")

# The NGC image sets this LD_LIBRARY_PATH, but processes launched from a
# Modal function didn't see it (TensorRT failed to load libnvonnxparser), so
# child processes get it explicitly.
NGC_LIBRARY_PATH = ":".join([
    "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu", "/opt/nvidia/nvda_nixl/lib64",
    "/usr/local/ucx/lib", "/usr/local/tensorrt/lib", "/usr/local/cuda/lib64",
    "/usr/local/lib/python3.12/dist-packages/torch/lib",
    "/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib",
    "/usr/local/cuda/compat/lib", "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64",
])


def _child_env() -> dict:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    return {"LD_LIBRARY_PATH": NGC_LIBRARY_PATH + (":" + current if current else "")}


VOL_PATH = "/vol"
MODEL_PATH = VOL_PATH + "/models/llama-3-8b-instruct"

image = add_repo_code(
    modal.Image.from_registry(TRTLLM_IMAGE)
    .entrypoint([])
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": VOL_PATH + "/hf"})
)

volume = modal.Volume.from_name("inferedge-trtllm", create_if_missing=True)
app = modal.App("inferedge-trtllm", image=image)

# Runs in a fresh Python process so every byte of GPU memory used while
# building is released before trtllm-serve starts.
BUILD_SCRIPT = r'''
import json, sys, time
from tensorrt_llm import BuildConfig
from tensorrt_llm._tensorrt_engine import LLM  # 1.x: tensorrt_llm.LLM is the PyTorch backend
from tensorrt_llm.llmapi import CalibConfig, QuantAlgo, QuantConfig

precision, model_path, engine_dir, max_batch, max_seq = sys.argv[1:6]
max_batch, max_seq = int(max_batch), int(max_seq)
build_config = BuildConfig(
    max_input_len=max_seq,
    max_seq_len=max_seq,
    max_num_tokens=8192,
    max_batch_size=max_batch,
)
kwargs = {}
if precision == "fp8":
    kwargs["quant_config"] = QuantConfig(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)
    kwargs["calib_config"] = CalibConfig(
        calib_batches=64, calib_batch_size=1, calib_max_seq_length=1024,
        tokenizer_max_seq_length=max_seq,
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
    engine_dir = "%s/engines/trtllm-1.2.1/%s-bs%d" % (VOL_PATH, precision, MAX_BATCH_SIZE)
    info_path = engine_dir + "/inferedge_build.json"
    cached = os.path.exists(info_path)
    if not cached:
        print("building %s engine (first run only)..." % precision, flush=True)
        print("function's own LD_LIBRARY_PATH: %r" % os.environ.get("LD_LIBRARY_PATH"), flush=True)
        subprocess.run(
            [sys.executable, "-c", BUILD_SCRIPT, precision, MODEL_PATH, engine_dir,
             str(MAX_BATCH_SIZE), str(MAX_SEQ_LEN)],
            check=True, env=dict(os.environ, **_child_env()),
        )
        volume.commit()
    with open(info_path) as f:
        info = json.load(f)
    info.update({"engine_dir": engine_dir, "engine_cached": cached})
    return info


def _serve_cmd(variant: str, build: dict) -> list:
    common_args = ["--host", "127.0.0.1", "--port", str(PORT),
                   "--max_batch_size", str(MAX_BATCH_SIZE),
                   "--kv_cache_free_gpu_memory_fraction", "0.90"]
    if variant == "pytorch-bf16":
        return ["trtllm-serve", "serve", MODEL_PATH, "--backend", "pytorch",
                "--max_seq_len", str(MAX_SEQ_LEN)] + common_args
    return ["trtllm-serve", "serve", build["engine_dir"], "--tokenizer", MODEL_PATH,
            "--backend", "tensorrt"] + common_args


@app.function(gpu=GPU, volumes={VOL_PATH: volume}, timeout=90 * MINUTES)
def sweep(variant: str = "bf16", smoke: bool = False) -> dict:
    from importlib.metadata import version

    from common import GpuSampler, Server, run_sweep_and_package

    if variant not in VARIANTS:
        raise ValueError("variant must be one of %s" % (VARIANTS,))
    _download_model()
    build = {"build_s": None, "engine_cached": None, "max_batch_size": MAX_BATCH_SIZE}
    if variant != "pytorch-bf16":
        build = _ensure_engine(variant)

    label = "trtllm-%s-%s%s" % (variant, GPU, "-smoke" if smoke else "")
    cmd = _serve_cmd(variant, build)
    with GpuSampler() as sampler:
        server = Server(cmd, env=_child_env())
        try:
            t0 = time.time()
            server.wait_ready()
            meta = {
                "engine": "TensorRT-LLM",
                "engine_version": version("tensorrt_llm"),
                "container": TRTLLM_IMAGE,
                "precision": "fp8" if variant == "fp8" else "bf16",
                "trtllm_backend": "pytorch" if variant == "pytorch-bf16" else "tensorrt",
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
def main(variants: str = "bf16,fp8,pytorch-bf16", smoke: bool = False):
    if smoke:
        variants = variants.split(",")[0]
    for variant in [v.strip() for v in variants.split(",") if v.strip()]:
        print("=== TensorRT-LLM %s on %s%s ===" % (variant, GPU, " (smoke)" if smoke else ""))
        try:
            save_outputs(sweep.remote(variant, smoke))
        except Exception as exc:  # keep going: one failed variant shouldn't lose the others
            print("TensorRT-LLM %s FAILED: %s" % (variant, exc))

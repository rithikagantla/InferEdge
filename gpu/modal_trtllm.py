"""TensorRT-LLM on one NVIDIA L4 (Modal): build engines, serve, sweep.

    modal run gpu/modal_trtllm.py --smoke                          # bf16 engine, c=1,2
    modal run gpu/modal_trtllm.py --variants bf16,fp8,pytorch-bf16  # engine vs PyTorch backend
    modal run gpu/modal_trtllm.py --variants pytorch-bf16-natural,spec-draft,spec-ngram
    modal run gpu/modal_trtllm.py --variants fp8-bs128,fp8-bs256,fp8-bs256-tok16k

All variants are served with `trtllm-serve` and measured with the same
sweep as vLLM (see VARIANTS below):

  Engine path      The model is compiled ahead of time into a TensorRT engine
                   specialized for this GPU (sm_89 / Ada), batch size and token
                   budget, then served with `--backend tensorrt`. fp8 adds
                   post-training quantization: NVIDIA ModelOpt runs calibration
                   batches through the bf16 model to pick FP8 scaling factors for
                   weights, activations and the KV cache.
  PyTorch backend  TensorRT-LLM 1.x's default: no engine build; the Hugging Face
                   checkpoint is served by TensorRT-LLM's PyTorch runtime. This is
                   also where speculative decoding is configured.
  Speculative      A cheap drafter proposes several tokens and the 8B model checks
                   them in one forward pass. `spec-draft` drafts with Llama 3.2 1B
                   (same 128,256-token vocabulary); `spec-ngram` drafts by matching
                   n-grams already in the prompt and output, with no second model.
                   With greedy decoding the output is identical to normal decoding,
                   so these run on natural-length answers against a matching
                   natural-length baseline.

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
MAX_SEQ_LEN = 2048

DRAFT_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"  # ungated; same vocabulary as Llama 3 8B
DRAFT_MODEL_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"

# name -> how to build/serve it. "engine" variants compile a TensorRT engine
# with the given precision, max batch size and max tokens per iteration.
VARIANTS = {
    "bf16": {"backend": "tensorrt", "precision": "bf16", "batch": 64, "tokens": 8192},
    "fp8": {"backend": "tensorrt", "precision": "fp8", "batch": 64, "tokens": 8192},
    "pytorch-bf16": {"backend": "pytorch", "precision": "bf16", "batch": 64},
    # Batching tuning on the fastest configuration (FP8 engine).
    "fp8-bs128": {"backend": "tensorrt", "precision": "fp8", "batch": 128, "tokens": 8192,
                  "levels": FULL_LEVELS + [128]},
    "fp8-bs256": {"backend": "tensorrt", "precision": "fp8", "batch": 256, "tokens": 8192,
                  "levels": FULL_LEVELS + [128, 256]},
    "fp8-bs256-tok16k": {"backend": "tensorrt", "precision": "fp8", "batch": 256, "tokens": 16384,
                         "levels": FULL_LEVELS + [128, 256]},
    # Speculative decoding, measured on natural-length answers.
    "pytorch-bf16-natural": {"backend": "pytorch", "precision": "bf16", "batch": 64,
                             "natural": True},
    # Speculation scores every draft position, so the runtime reserves a
    # float32 logits buffer of max_num_tokens x 128,256 vocab: 3.9 GiB at the
    # default 8,192 tokens, which didn't fit next to the KV cache. These
    # prompts never need more than ~320 tokens per step (64 requests x 5
    # positions), so 2,048 loses nothing and cuts that buffer to ~1 GiB.
    # The YAML key is "DraftTarget" even though the class's own decoding_type
    # attribute says "Draft_Target".
    # Draft-target also holds a second model (2.5 GB) and its logits buffers:
    # 8B + 1B weights leave ~3.5 GB on the L4, so the KV cache gets 35% of that
    # and the per-step budget 1,024 tokens.
    # It also crashed in the overlap scheduler (assert request.py_is_first_draft
    # in _apply_incremental_update_draft), so the overlap scheduler is off.
    "spec-draft": {"backend": "pytorch", "precision": "bf16", "batch": 64, "natural": True,
                   "kv_fraction": 0.35, "tokens": 1024,
                   "llm_options": {"disable_overlap_scheduler": True},
                   "speculative": {"decoding_type": "DraftTarget", "max_draft_len": 4}},
    "spec-ngram": {"backend": "pytorch", "precision": "bf16", "batch": 64, "natural": True,
                   "kv_fraction": 0.70, "tokens": 2048,
                   "speculative": {"decoding_type": "NGram", "max_draft_len": 4,
                                   "max_matching_ngram_size": 3}},
    # Default NGram shares one n-gram pool across requests. With repeated
    # prompts and greedy decoding, answers repeat word for word, so the shared
    # pool turns into an answer cache. A private pool drafts only from the
    # request's own prompt and output: the honest speculative-decoding number.
    "spec-ngram-private": {"backend": "pytorch", "precision": "bf16", "batch": 64, "natural": True,
                           "kv_fraction": 0.70, "tokens": 2048,
                           "speculative": {"decoding_type": "NGram", "max_draft_len": 4,
                                           "max_matching_ngram_size": 3,
                                           "is_public_pool": False}},
}

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
DRAFT_PATH = VOL_PATH + "/models/llama-3.2-1b-instruct"

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

precision, model_path, engine_dir, max_batch, max_seq, max_tokens = sys.argv[1:7]
max_batch, max_seq, max_tokens = int(max_batch), int(max_seq), int(max_tokens)
build_config = BuildConfig(
    max_input_len=max_seq,
    max_seq_len=max_seq,
    max_num_tokens=max_tokens,
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
    json.dump({"precision": precision, "build_s": round(build_s, 1), "max_batch_size": max_batch,
               "max_num_tokens": max_tokens}, f)
print("ENGINE BUILT in %.0fs" % build_s, flush=True)
'''


def _download(model_id: str, revision: str, path: str) -> None:
    from huggingface_hub import snapshot_download

    if os.path.exists(path + "/config.json"):
        return
    snapshot_download(model_id, local_dir=path, revision=revision,
                      ignore_patterns=["*.pt", "*.bin", "original/*"])
    volume.commit()


def _ensure_engine(spec: dict) -> dict:
    engine_dir = "%s/engines/trtllm-1.2.1/%s-bs%d%s" % (
        VOL_PATH, spec["precision"], spec["batch"],
        "" if spec["tokens"] == 8192 else "-tok%d" % spec["tokens"],
    )
    info_path = engine_dir + "/inferedge_build.json"
    cached = os.path.exists(info_path)
    if not cached:
        print("building %s engine (first run only)..." % os.path.basename(engine_dir), flush=True)
        print("function's own LD_LIBRARY_PATH: %r" % os.environ.get("LD_LIBRARY_PATH"), flush=True)
        subprocess.run(
            [sys.executable, "-c", BUILD_SCRIPT, spec["precision"], MODEL_PATH, engine_dir,
             str(spec["batch"]), str(MAX_SEQ_LEN), str(spec["tokens"])],
            check=True, env=dict(os.environ, **_child_env()),
        )
        volume.commit()
    with open(info_path) as f:
        info = json.load(f)
    info.update({"engine_dir": engine_dir, "engine_cached": cached})
    return info


def _serve_cmd(spec: dict, build: dict) -> list:
    common_args = ["--host", "127.0.0.1", "--port", str(PORT),
                   "--max_batch_size", str(spec["batch"]),
                   "--kv_cache_free_gpu_memory_fraction", str(spec.get("kv_fraction", 0.90))]
    if spec["backend"] == "pytorch":
        cmd = ["trtllm-serve", "serve", MODEL_PATH, "--backend", "pytorch",
               "--max_seq_len", str(MAX_SEQ_LEN)] + common_args
        if spec.get("tokens"):
            cmd += ["--max_num_tokens", str(spec["tokens"])]
        if spec.get("speculative"):
            # trtllm-serve takes speculative decoding settings via a YAML file
            # of extra LLM API options (JSON is valid YAML).
            options = {"speculative_config": dict(spec["speculative"])}
            options.update(spec.get("llm_options", {}))
            if options["speculative_config"]["decoding_type"] == "DraftTarget":
                options["speculative_config"]["speculative_model"] = DRAFT_PATH
            path = "/tmp/extra_llm_api_options.yaml"
            with open(path, "w") as f:
                json.dump(options, f)
            cmd += ["--extra_llm_api_options", path]
        return cmd
    return ["trtllm-serve", "serve", build["engine_dir"], "--tokenizer", MODEL_PATH,
            "--backend", "tensorrt"] + common_args


@app.function(gpu=GPU, volumes={VOL_PATH: volume}, timeout=90 * MINUTES)
def sweep(variant: str = "bf16", smoke: bool = False) -> dict:
    from importlib.metadata import version

    from common import GpuSampler, Server, run_sweep_and_package

    if variant not in VARIANTS:
        raise ValueError("variant must be one of %s" % sorted(VARIANTS))
    spec = VARIANTS[variant]
    _download(MODEL_ID, MODEL_REVISION, MODEL_PATH)
    if (spec.get("speculative") or {}).get("decoding_type") == "DraftTarget":
        _download(DRAFT_MODEL_ID, DRAFT_MODEL_REVISION, DRAFT_PATH)
    build = {"build_s": None, "engine_cached": None, "max_batch_size": spec["batch"],
             "max_num_tokens": spec.get("tokens")}
    if spec["backend"] == "tensorrt":
        build = _ensure_engine(spec)

    label = "trtllm-%s-%s%s" % (variant, GPU, "-smoke" if smoke else "")
    cmd = _serve_cmd(spec, build)
    with GpuSampler() as sampler:
        server = Server(cmd, env=_child_env())
        try:
            t0 = time.time()
            server.wait_ready()
            meta = {
                "engine": "TensorRT-LLM",
                "engine_version": version("tensorrt_llm"),
                "container": TRTLLM_IMAGE,
                "variant": variant,
                "precision": spec["precision"],
                "trtllm_backend": spec["backend"],
                "speculative_config": spec.get("speculative"),
                "draft_model": DRAFT_MODEL_ID if (spec.get("speculative") or {}).get(
                    "decoding_type") == "DraftTarget" else None,
                "hf_model": MODEL_ID,
                "hf_revision": MODEL_REVISION,
                "engine_build_s": build["build_s"],
                "engine_cached": build["engine_cached"],
                "max_batch_size": build["max_batch_size"],
                "max_num_tokens": build.get("max_num_tokens"),
                "kv_cache_free_gpu_memory_fraction": spec.get("kv_fraction", 0.90),
                "server_startup_s": round(time.time() - t0, 1),
                "server_cmd": " ".join(cmd),
            }
            levels = SMOKE_LEVELS if smoke else spec.get("levels", FULL_LEVELS)
            return run_sweep_and_package(
                label, levels, meta, server, sampler,
                ignore_eos=False if spec.get("natural") else None,
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

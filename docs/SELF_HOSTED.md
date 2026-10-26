# Self-hosted GPU benchmarks (vLLM and TensorRT-LLM on an NVIDIA L4)

This is how the throughput-vs-latency numbers in the README were measured, and how to reproduce them. Everything runs on [Modal](https://modal.com), which bills GPUs per second. The Starter plan includes $30/month of free credits, and an L4 costs about $0.80/hour, so a full set of runs fits comfortably inside the free credits.

## What gets measured

`scripts/sweep_serving.py` drives an OpenAI-compatible server with **closed-loop load**. At concurrency *C*, *C* workers each send requests back to back, so the server always has *C* requests in flight, and its continuous batcher runs at a batch size of about *C*. Each level (1, 2, 4, 8, 16, 32, 64) records:

| Metric | Meaning |
|---|---|
| TTFT | Time to first token: queueing plus prefill. What a user waits before text appears. |
| TPOT | Time per output token after the first, `(e2e − TTFT) / (tokens − 1)`. How fast text streams for one user. |
| ITL | Inter-token latency: the gaps between streamed chunks (p50 and p99). |
| Output tok/s | Generated tokens per second across all requests: the server's throughput. |
| Peak GPU memory | Sampled from `nvidia-smi` while serving. |

The sweep runs **inside the GPU container against localhost**, so internet round trips don't leak into TTFT or ITL. Every request generates exactly 256 tokens (`ignore_eos`), so each level does comparable work. The engines also share one streaming client (`backend/app/inference/openai_compat.py`), so vLLM, TensorRT-LLM and hosted NIM are all measured the same way.

## Setup (one time)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv isn't installed
uv tool install modal --python 3.12                 # Modal's client needs Python 3.10–3.14
modal setup                                         # browser login
```

## Run

```bash
modal run gpu/modal_vllm.py --smoke                 # ~5 min sanity check
modal run gpu/modal_vllm.py --variants bf16,fp8,fp8-kv   # vLLM sweeps (fp8-kv: FP8 KV cache too)
modal run gpu/modal_trtllm.py --variants bf16,fp8,pytorch-bf16  # TensorRT-LLM: builds engines on first run
modal run gpu/modal_trtllm.py --variants pytorch-bf16-natural,spec-draft,spec-ngram-private  # speculative decoding
modal run gpu/modal_trtllm.py --variants fp8-bs128,fp8-bs256,fp8-bs256-tok16k  # batching limits
modal run gpu/modal_router.py                       # semantic router: CPU vs GPU (CuPy) scoring
modal run gpu/modal_router_embed.py                 # router with MiniLM embeddings: accuracy + GPU latency
backend/.venv/bin/pip install -r scripts/requirements.txt
backend/.venv/bin/python scripts/plot_sweep.py benchmark_results/gpu/sweep_*-L4.json
```

Results are written to `benchmark_results/gpu/sweep_<label>.{json,csv,md}`. Each JSON also records the engine version, the server command, startup and engine build time, peak GPU memory, and the server's own log lines about how memory was split between weights and KV cache.

## Use the self-hosted model from the InferEdge app

```bash
modal serve gpu/modal_vllm.py      # prints a URL; GPU stops 5 min after the last request
```

Then set this in `.env` and restart the backend:

```bash
INFERENCE_MODE=local
LOCAL_BASE_URL=https://<printed-url>/v1
LOCAL_MODEL=NousResearch/Meta-Llama-3-8B-Instruct
```

`scripts/run_benchmark.py` and the dashboard then run against your own GPU. Press Ctrl+C on `modal serve` when you're done so the GPU stops billing.

## Model and versions

- **Model:** `NousResearch/Meta-Llama-3-8B-Instruct` at revision `53346005`, an ungated copy of Meta's Llama 3 8B Instruct, so no Hugging Face token is needed.
- **vLLM 0.21.0:** the version pinned in Modal's maintained example.
- **TensorRT-LLM 1.2.1:** NVIDIA's prebuilt `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` container from NGC. The 0.20.0 version in Modal's original example can no longer be pip-installed. See [TENSORRT_LLM.md](TENSORRT_LLM.md).
- **GPU:** one NVIDIA L4 (Ada Lovelace, 24 GB, about 300 GB/s memory bandwidth, native FP8 tensor cores).

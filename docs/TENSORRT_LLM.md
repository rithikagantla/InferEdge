# TensorRT-LLM integration: what I built, measured, and learned

This is a working log of integrating NVIDIA TensorRT-LLM into InferEdge as a self-hosted serving backend, next to vLLM, on a single NVIDIA L4. The integration is partial and in progress: the engines build, serve, and are benchmarked, and the next steps are listed at the end.

## What is integrated

`gpu/modal_trtllm.py` runs on a cloud L4 through Modal and does four things:

1. **Downloads the model** (`NousResearch/Meta-Llama-3-8B-Instruct`, an ungated copy of Llama 3 8B Instruct) into a persistent volume.
2. **Builds TensorRT engines ahead of time.** Unlike vLLM, which loads the checkpoint and runs it, TensorRT-LLM compiles the model into an engine specialized for this exact GPU (sm_89, Ada), batch size (64), and sequence length (2,048). Engines are cached, so each build happens once.
3. **Serves the engine** with `trtllm-serve`, which exposes the same OpenAI-compatible API as vLLM and hosted NIM.
4. **Measures it** with `scripts/sweep_serving.py`, the same load generator and streaming client used for vLLM and hosted NIM, so every number in the comparison comes from the same code path.

Because `trtllm-serve` is OpenAI-compatible, the InferEdge app itself can also use it through `INFERENCE_MODE=local` (`backend/app/inference/local_engine.py`) with no other changes.

Three variants are measured:

| Variant | How it runs |
|---|---|
| `bf16` | Compiled TensorRT engine in the checkpoint's native precision |
| `fp8` | Compiled engine with post-training FP8 quantization of weights, activations and KV cache. NVIDIA ModelOpt runs calibration batches through the bf16 model to choose FP8 scaling factors before the build. |
| `pytorch-bf16` | TensorRT-LLM 1.x's new default backend: no engine build; the Hugging Face checkpoint is served by TensorRT-LLM's PyTorch runtime with its optimized kernels |

## Results

Measured on October 23, 2026 on one NVIDIA L4 (24 GB) with the same sweep as vLLM: 1 to 64 concurrent requests, closed-loop load, exactly 256 output tokens per request. "1 user" is concurrency 1; "64 users" is concurrency 64. TTFT is time to first token, and ITL p99 is the worst 1% of gaps between streamed tokens.

| Engine | Precision | 1 user: ms per token | 1 user: TTFT (ms) | 64 users: total tok/s | 64 users: TTFT p50 (ms) | 64 users: ITL p99 (ms) | KV cache (tokens) |
|---|---|---:|---:|---:|---:|---:|---:|
| TensorRT-LLM engine | FP8 (weights + KV cache) | 38.3 | 44 | 1,448 | 131 | 48 | 170,176 |
| vLLM | FP8 (weights) | 35.7 | 51 | 1,332 | 298 | 150 | 81,008 |
| TensorRT-LLM engine | bf16 | 61.5 | 70 | 889 | 199 | 223 | 39,552 |
| vLLM | bf16 | 59.1 | 78 | 851 | 367 | 228 | 34,384 |
| TensorRT-LLM PyTorch backend | bf16 | 59.4 | 78 | 837 | 289 | 236 | 39,424 |

Output throughput (tokens/s, all requests) at each concurrency level:

| Concurrency | trtllm-fp8 | vllm-fp8 | trtllm-bf16 | vllm-bf16 | trtllm-pytorch-bf16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 26 | 28 | 16 | 17 | 17 |
| 2 | 53 | 55 | 32 | 33 | 32 |
| 4 | 105 | 110 | 64 | 65 | 64 |
| 8 | 206 | 215 | 126 | 128 | 127 |
| 16 | 403 | 419 | 248 | 251 | 246 |
| 32 | 776 | 776 | 479 | 452 | 443 |
| 64 | 1,448 | 1,332 | 889 | 851 | 837 |

![Throughput vs latency on an L4](img/throughput_vs_latency_l4.png)

**What the numbers say:**

- **FP8 is the biggest lever, in both engines.** One user's text streams about 1.6x faster (59 → 36 ms per token in vLLM, 62 → 38 in TensorRT-LLM), and peak throughput rises 56–63%. Decoding is memory-bound: every token requires reading all the weights, so halving their size nearly halves the time per token.
- **TensorRT-LLM wins under load; vLLM wins one user at a time.** With a single user, vLLM was slightly faster per token in both precisions (by 2–3 ms). At 64 concurrent users, the TensorRT-LLM FP8 engine delivered 9% more throughput, less than half of vLLM's time to first token (131 vs 298 ms), and much steadier streaming (48 vs 150 ms worst-case gap between tokens).
- **FP8 KV cache is the hidden win.** The TensorRT-LLM FP8 engine also stores its KV cache in FP8, so it fit 170,176 tokens of cache, 4.3x its bf16 engine and 2.1x vLLM's FP8 mode, which keeps the cache in bf16. More cache means more concurrent conversations before requests must wait.
- **The compiled engine beat TensorRT-LLM's own default.** In bf16 the ahead-of-time engine reached 889 tok/s with a 199 ms first-token wait at 64 users, while the 1.x default PyTorch backend reached 837 tok/s and 289 ms, about the same as vLLM. The engine pays for that with a 3-minute build per configuration.
- **No errors** in any of the 1,000+ requests across the three variants.

## What broke, and what I learned

**1. The published recipe no longer installs.** I started from Modal's TensorRT-LLM example, which pins `tensorrt-llm==0.20.0`. That version requires exactly `xgrammar==0.1.16`, which has since been removed from PyPI, so `pip install tensorrt-llm==0.20.0` now fails for everyone. The lesson: pin the whole environment, not just the top-level package. The fix was NVIDIA's prebuilt `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` container from NGC, which ships every dependency already resolved. That is NVIDIA's recommended way to run TensorRT-LLM anyway.

**2. The API moved between 0.20 and 1.x.** In 1.x, `tensorrt_llm.LLM` is the PyTorch backend. The classic build-an-engine path moved to `tensorrt_llm._tensorrt_engine.LLM`, and serving an engine needs `trtllm-serve serve <engine> --backend tensorrt`. Rather than guess, I started the container on a GPU and read `trtllm-serve serve --help` and the module layout directly.

**3. The container's library path was silently replaced.** The engine build failed with `ImportError: libnvonnxparser.so.10`, even though the file exists in the image at `/usr/local/tensorrt/lib`. An interactive shell in the same image had that directory on `LD_LIBRARY_PATH`, but processes launched from the Modal function did not: the runtime's `LD_LIBRARY_PATH` kept only the PyTorch and CUDA entries. Passing NVIDIA's full library path to the build and server processes fixed it. The lesson: when "the file is right there" but loading fails, compare the environment of the failing process, not the image.

**4. My first L4 run used the wrong CUDA image for CuPy.** (Router benchmark, same session.) CuPy compiles its reduction kernels at run time with NVRTC, which needs the CUDA headers that only the `devel` images include. With the slim `runtime` image the compile error was printed as "Exception ignored" and timing continued, producing numbers for a kernel that never ran. The benchmark now refuses to time anything until the GPU and CPU agree on every result.

**5. Calibrated FP8 was cheap to build.** FP8 post-training quantization with 64 calibration batches built in 182 seconds on the L4, about the same as the bf16 engine (192 seconds).

**6. "Fastest engine" depends on the load.** A single-request benchmark would have crowned vLLM; a throughput benchmark crowns TensorRT-LLM. For a real-time support agent, the deciding numbers are first-token wait and worst-case streaming gaps at realistic concurrency, which is why the sweep records both.

## Next steps

- **Speculative decoding.** TensorRT-LLM supports draft-model, Medusa and EAGLE speculative decoding. At low concurrency, where the L4 spends most of its time waiting on memory, guessing several tokens per step and verifying them in one pass should cut time per output token substantially.
- **In-flight batching tuning.** Sweep `max_batch_size`, `max_num_tokens` and the KV cache fraction rather than using one configuration, and try chunked prefill to keep first-token latency flat under load.
- **FP8 KV cache in vLLM** for a like-for-like comparison: the TensorRT-LLM FP8 engine quantizes the KV cache too, while vLLM's `--quantization fp8` does not.
- **Triton Inference Server or NVIDIA Dynamo** in front of the engines, for production batching policies and KV-cache-aware routing across replicas, which is the production version of InferEdge's semantic router.

## Reproduce

```bash
modal run gpu/modal_trtllm.py --variants bf16,fp8,pytorch-bf16
backend/.venv/bin/python scripts/plot_sweep.py benchmark_results/gpu/sweep_*-L4.json
```

See [SELF_HOSTED.md](SELF_HOSTED.md) for setup.

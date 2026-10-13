"""Helpers shared by the Modal GPU scripts (gpu/modal_vllm.py, gpu/modal_trtllm.py).

Everything here runs inside the GPU container except save_outputs(),
which the local entrypoint uses to write results on your machine.
"""
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/inferedge"

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"  # ungated copy of Llama 3 8B Instruct
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
GPU = "L4"
PORT = 8000

# Server log lines worth keeping: how memory was split between weights
# and KV cache, and how many concurrent sequences that cache can hold.
_INTERESTING = re.compile(
    r"(Model loading took|KV cache|Maximum concurrency|memory|Memory|max_batch_size|max_num_tokens)"
)


def add_repo_code(image):
    """Ship the InferEdge client code (shared streaming client + sweep) into the image."""
    return (
        image.add_local_dir(REPO_ROOT / "backend" / "app", REMOTE_ROOT + "/backend/app",
                            ignore=["**/__pycache__/**"])
        .add_local_dir(REPO_ROOT / "scripts", REMOTE_ROOT + "/scripts",
                       ignore=["**/__pycache__/**"])
        .add_local_python_source("common")
    )


class Server:
    """Runs an OpenAI-compatible server as a subprocess and keeps its log."""

    def __init__(self, cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
        print("$ " + " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=dict(os.environ, **(env or {})),
        )
        self.tail: deque = deque(maxlen=80)
        self.notes: List[str] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            self.tail.append(line)
            if _INTERESTING.search(line) and len(self.notes) < 40:
                self.notes.append(line[-300:])

    def wait_ready(self, timeout_s: float = 1500) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "server exited with %s:\n%s" % (self.proc.returncode, "\n".join(self.tail))
                )
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(3)
        raise TimeoutError("server not healthy after %ds" % timeout_s)

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class GpuSampler:
    """Polls nvidia-smi in the background; records GPU name and peak memory."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.name = None
        self.total_mib = None
        self.peak_used_mib = 0
        self.peak_util_pct = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=10,
                ).strip().splitlines()[0]
                name, used, total, util = [x.strip() for x in out.split(",")]
                self.name, self.total_mib = name, int(total)
                self.peak_used_mib = max(self.peak_used_mib, int(used))
                self.peak_util_pct = max(self.peak_util_pct, int(util))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict:
        return {
            "gpu_name": self.name,
            "gpu_memory_total_mib": self.total_mib,
            "gpu_memory_peak_used_mib": self.peak_used_mib,
            "gpu_util_peak_pct": self.peak_util_pct,
        }


def served_model_id() -> str:
    import json

    with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def run_sweep_and_package(label: str, levels: List[int], meta: dict, server: Server,
                          sampler: GpuSampler) -> Dict[str, str]:
    """Run the InferEdge sweep against the local server; return output files as text."""
    import asyncio
    import tempfile

    sys.path.insert(0, REMOTE_ROOT + "/scripts")
    import sweep_serving

    model = served_model_id()
    base_url = "http://127.0.0.1:%d/v1" % PORT
    ignore_eos = _supports_ignore_eos(sweep_serving, base_url, model)
    result = asyncio.run(sweep_serving.run_sweep(
        base_url, model, label, levels=levels, ignore_eos=ignore_eos,
    ))
    result.update(meta)
    result.update(sampler.summary())
    result["server_log_notes"] = server.notes

    out_dir = tempfile.mkdtemp()
    base = sweep_serving.write_outputs(result, out_dir)
    return {Path(base + ext).name: Path(base + ext).read_text() for ext in (".json", ".csv", ".md")}


def _supports_ignore_eos(sweep_serving, base_url: str, model: str) -> bool:
    """Probe once: some servers reject the ignore_eos extension."""
    import asyncio

    probe = asyncio.run(sweep_serving.run_sweep(
        base_url, model, "probe", levels=[1], max_tokens=8, ignore_eos=True,
        min_requests=1, per_worker=1, warmup=0, log=lambda *_: None,
    ))["levels"][0]
    if probe["errors"]:
        print("ignore_eos rejected, measuring natural-length outputs: %s"
              % probe.get("first_error"), flush=True)
        return False
    return True


def save_outputs(files: Dict[str, str], out_dir: str = "benchmark_results/gpu") -> None:
    out = REPO_ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
        print("saved", out / name)
    md = [t for n, t in files.items() if n.endswith(".md")]
    if md:
        print("\n" + md[0])

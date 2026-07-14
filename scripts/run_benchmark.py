#!/usr/bin/env python3
"""InferEdge benchmark CLI.

Runs N customer-support prompts through every optimization mode via the
InferEdge API, saves JSON + CSV results, and prints a summary table.

Usage:
    python scripts/run_benchmark.py                     # 50 prompts, all modes
    python scripts/run_benchmark.py --num-prompts 200
    python scripts/run_benchmark.py --modes baseline batching caching
    python scripts/run_benchmark.py --base-url http://localhost:8000

The backend must be running first:
    cd backend && .venv/bin/uvicorn app.main:app --port 8000
"""
import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime

DEFAULT_BASE_URL = os.getenv("INFEREDGE_URL", "http://127.0.0.1:8000")
ALL_MODES = ["baseline", "batching", "caching", "routing", "quantized"]

CSV_COLUMNS = [
    "mode", "num_requests", "wall_time_s", "requests_per_sec",
    "p50_latency_ms", "p95_latency_ms", "avg_ttft_ms", "avg_tokens_per_sec",
    "total_input_tokens", "total_output_tokens", "total_cost_usd",
    "est_cost_per_1m_tokens", "cache_hit_rate",
]


def post_json(url: str, payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fmt(value, width, decimals=2):
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:.{decimals}f}".rjust(width)
    return str(value).rjust(width)


def print_table(results: list) -> None:
    header = (
        f"{'mode':<10} {'reqs':>5} {'wall(s)':>8} {'req/s':>7} "
        f"{'p50(ms)':>8} {'p95(ms)':>8} {'ttft(ms)':>9} {'tok/s':>8} "
        f"{'$ total':>10} {'$/1M tok':>9} {'cache':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        hit_rate = r.get("cache_hit_rate")
        print(
            f"{r['mode']:<10} {r['num_requests']:>5} {fmt(r['wall_time_s'], 8)} "
            f"{fmt(r['requests_per_sec'], 7)} {fmt(r['p50_latency_ms'], 8)} "
            f"{fmt(r['p95_latency_ms'], 8)} {fmt(r['avg_ttft_ms'], 9)} "
            f"{fmt(r['avg_tokens_per_sec'], 8)} {fmt(r['total_cost_usd'], 10, 6)} "
            f"{fmt(r['est_cost_per_1m_tokens'], 9, 4)} "
            f"{fmt(hit_rate * 100 if hit_rate is not None else None, 5, 1)}%"
        )
    print()
    baseline = next((r for r in results if r["mode"] == "baseline"), None)
    if baseline:
        for r in results:
            if r["mode"] == "baseline":
                continue
            speedup = baseline["p50_latency_ms"] / r["p50_latency_ms"] if r["p50_latency_ms"] else float("inf")
            thrpt = r["requests_per_sec"] / baseline["requests_per_sec"] if baseline["requests_per_sec"] else 0
            cost = (
                (1 - r["est_cost_per_1m_tokens"] / baseline["est_cost_per_1m_tokens"]) * 100
                if baseline["est_cost_per_1m_tokens"]
                else 0
            )
            print(
                f"  {r['mode']:<10} vs baseline:  {speedup:5.2f}x p50 speedup, "
                f"{thrpt:5.2f}x throughput, {cost:+5.1f}% cost savings per token"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="InferEdge benchmark runner")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--num-prompts", type=int, default=50, choices=range(1, 201),
                        metavar="[1-200]")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    parser.add_argument("--out-dir", default="benchmark_results")
    parser.add_argument("--router-intents", type=int, default=5000,
                        help="intents for the CPU-vs-GPU router benchmark")
    args = parser.parse_args()

    print(f"InferEdge benchmark — {args.num_prompts} prompts, modes: {', '.join(args.modes)}")
    print(f"API: {args.base_url}")

    try:
        info = json.loads(urllib.request.urlopen(args.base_url + "/", timeout=5).read())
    except (urllib.error.URLError, OSError):
        print(
            "\nERROR: cannot reach the InferEdge API. Start it first:\n"
            "  cd backend && .venv/bin/uvicorn app.main:app --port 8000",
            file=sys.stderr,
        )
        return 1
    print(f"Inference mode: {info['inference_mode']}  |  Router: {info['router_backend']}")

    print("\nRunning inference benchmark (this can take a minute)...")
    run = post_json(
        args.base_url + "/benchmark/run",
        {"num_prompts": args.num_prompts, "modes": args.modes},
    )
    print_table(run["results"])

    print(f"Running semantic-router CPU vs GPU benchmark ({args.router_intents} intents)...")
    router_bench = post_json(
        args.base_url + "/router/benchmark", {"num_intents": args.router_intents}
    )
    print(f"  CPU (NumPy): {router_bench['cpu_time_ms']:.3f} ms")
    if router_bench["gpu_available"]:
        print(f"  GPU (CuPy):  {router_bench['gpu_time_ms']:.3f} ms")
        print(f"  Speedup:     {router_bench['speedup']:.2f}x")
    else:
        print("  GPU: not available — " + router_bench["note"])

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out_dir, f"benchmark_{stamp}.json")
    csv_path = os.path.join(args.out_dir, f"benchmark_{stamp}.csv")

    with open(json_path, "w") as f:
        json.dump({"benchmark": run, "router_benchmark": router_bench}, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(run["results"])

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Plot throughput-vs-latency curves from sweep JSON files.

    python scripts/plot_sweep.py benchmark_results/gpu/sweep_*.json \
        --out docs/img/throughput_vs_latency_l4.png

Left panel: output tokens/s vs median time per output token (the latency
each user feels while text streams). Right panel: output tokens/s vs
median time to first token. Each point is one concurrency level; moving
along a curve trades per-user latency for total throughput.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed order, colorblind-safe (Okabe-Ito); identity is also carried by
# marker shape and direct labels, never color alone.
SERIES_STYLE = [
    ("#0072B2", "o"),
    ("#E69F00", "s"),
    ("#009E73", "^"),
    ("#CC79A7", "D"),
    ("#56B4E9", "v"),
    ("#D55E00", "P"),
]
INK = "#1f2328"
MUTED = "#59636e"
GRID = "#d8dee4"


def load(paths):
    sweeps = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        levels = [lv for lv in data["levels"] if lv.get("requests") and lv.get("tpot_p50_ms")]
        if levels:
            data["levels"] = sorted(levels, key=lambda lv: lv["concurrency"])
            sweeps.append(data)
    return sweeps


def panel(ax, sweeps, x_key, x_label):
    series_labels = []
    for i, sw in enumerate(sweeps):
        color, marker = SERIES_STYLE[i % len(SERIES_STYLE)]
        xs = [lv[x_key] for lv in sw["levels"]]
        ys = [lv["output_tokens_per_sec"] for lv in sw["levels"]]
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=sw["label"], zorder=3)
        # Label every concurrency level when there are few points; with many
        # series only the top level, since the low levels bunch together.
        if len(sweeps) > 4:
            picks = set()  # the footnote says which end is which
        elif len(sweeps) > 3:
            picks = {len(xs) - 1}
        else:
            picks = set(range(len(xs))) if len(xs) <= 5 else {0, len(xs) // 2, len(xs) - 1}
        for j, lv in enumerate(sw["levels"]):
            if j in picks:
                ax.annotate("c=%d" % lv["concurrency"], (xs[j], ys[j]), textcoords="offset points",
                            xytext=(7, -4), fontsize=8, color=MUTED, va="center")
        series_labels.append((sw["label"], xs, ys, color))
    ax.set_xlabel(x_label, color=INK)
    ax.set_ylabel("Output throughput (tokens/s, all requests)", color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_ylim(bottom=0, top=max(max(y) for _, _, y, _ in series_labels) * 1.15)
    x_min = min(min(x) for _, x, _, _ in series_labels)
    x_max = max(max(x) for _, x, _, _ in series_labels)
    if x_max / x_min > 20:
        # One curve's queueing blow-up would squash the rest into a sliver.
        ax.set_xscale("log")
        ax.set_xlabel(ax.get_xlabel() + "  (log scale)", color=INK)
        ax.set_xlim(left=x_min * 0.8, right=x_max * 1.6)
    else:
        ax.set_xlim(left=x_min * 0.85, right=x_max * 1.12)
    # More than four series: the legend carries identity and direct labels
    # would collide.
    if len(series_labels) > 4:
        return
    # Direct label above each curve's highest-throughput point, aligned
    # inward so it never runs off the plot.
    lo, hi = ax.get_xlim()
    x_mid = (lo * hi) ** 0.5 if ax.get_xscale() == "log" else (lo + hi) / 2
    for label, xs, ys, color in series_labels:
        j = ys.index(max(ys))
        ax.annotate(label, (xs[j], ys[j]), textcoords="offset points", xytext=(0, 9),
                    ha="right" if xs[j] > x_mid else "left", fontsize=9, color=INK,
                    fontweight="bold")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="docs/img/throughput_vs_latency_l4.png")
    parser.add_argument("--title", default="Llama-3-8B-Instruct on one NVIDIA L4: throughput vs latency")
    args = parser.parse_args()

    sweeps = load(args.paths)
    if not sweeps:
        print("no sweep data with completed levels", file=sys.stderr)
        return 1

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    panel(left, sweeps, "tpot_p50_ms", "Median time per output token (ms)  → slower streaming")
    panel(right, sweeps, "ttft_p50_ms", "Median time to first token (ms)  → longer wait")
    left.legend(frameon=False, loc="upper left" if len(sweeps) > 4 else "best", fontsize=9)
    fig.suptitle(args.title, color=INK, fontsize=13, fontweight="bold", x=0.01, ha="left")
    if all(sw.get("ignore_eos") for sw in sweeps):
        length = "%d output tokens per request" % sweeps[0].get("max_tokens", 256)
    else:
        length = "natural-length answers, up to %d tokens" % max(sw.get("max_tokens", 256) for sw in sweeps)
    levels = sorted({lv["concurrency"] for sw in sweeps for lv in sw["levels"]})
    rising = all(
        all(a["output_tokens_per_sec"] < b["output_tokens_per_sec"]
            for a, b in zip(sw["levels"], sw["levels"][1:]))
        for sw in sweeps
    )
    if rising:
        where = "Each curve runs from %d concurrent request (bottom) to %d (top)" % (levels[0], levels[-1])
    else:
        where = "Points are concurrency levels %d to %d (labeled c=N)" % (levels[0], levels[-1])
    fig.text(0.01, 0.005, "%s; closed-loop load, %s. Up and to the left is better."
             % (where, length), fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print("Saved", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

import { useEffect, useState } from "react";
import { api } from "../api";
import type { BenchmarkResult } from "../types";
import BarChart from "./BarChart";

export default function BenchmarkPanel({ onActivity }: { onActivity: () => void }) {
  const [numPrompts, setNumPrompts] = useState(20);
  const [run, setRun] = useState<BenchmarkResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Show the most recent saved run on load.
  useEffect(() => {
    api
      .benchmarkResults()
      .then((d) => d.runs.length && setRun(d.runs[0]))
      .catch(() => {});
  }, []);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      setRun(await api.runBenchmark(numPrompts));
      onActivity();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const results = run?.results ?? [];
  return (
    <div className="card full">
      <h2>Optimization benchmark</h2>
      <p className="hint">
        Runs the sample-prompt suite through every optimization mode and
        compares latency, throughput, and cost per token.
      </p>
      <div className="row">
        <select
          value={numPrompts}
          onChange={(e) => setNumPrompts(Number(e.target.value))}
          aria-label="Number of prompts"
        >
          {[10, 20, 50, 100].map((n) => (
            <option key={n} value={n}>
              {n} prompts
            </option>
          ))}
        </select>
        <button onClick={start} disabled={busy}>
          {busy ? "Running… (up to a minute)" : "Run benchmark"}
        </button>
        {run && !busy && (
          <span className="spin">
            last run: {new Date(run.created_at).toLocaleString()} ·{" "}
            {run.num_prompts} prompts · {run.inference_mode} mode
          </span>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      {results.length > 0 && (
        <>
          <div className="grid" style={{ marginTop: 8 }}>
            <BarChart
              title="p50 latency (ms) — lower is better"
              bars={results.map((r) => ({ name: r.mode, value: r.p50_latency_ms }))}
              format={(v) => `${v.toFixed(1)}`}
            />
            <BarChart
              title="Throughput (requests/sec) — higher is better"
              bars={results.map((r) => ({ name: r.mode, value: r.requests_per_sec }))}
            />
            <BarChart
              title="p95 latency (ms) — lower is better"
              bars={results.map((r) => ({ name: r.mode, value: r.p95_latency_ms }))}
              format={(v) => `${v.toFixed(1)}`}
            />
            <BarChart
              title="Est. cost per 1M tokens ($) — lower is better"
              bars={results.map((r) => ({
                name: r.mode,
                value: r.est_cost_per_1m_tokens,
              }))}
              format={(v) => `$${v.toFixed(3)}`}
            />
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>mode</th>
                  <th>reqs</th>
                  <th>wall (s)</th>
                  <th>req/s</th>
                  <th>p50 (ms)</th>
                  <th>p95 (ms)</th>
                  <th>ttft (ms)</th>
                  <th>tok/s</th>
                  <th>cost ($)</th>
                  <th>$/1M tok</th>
                  <th>cache</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.mode} title={r.notes ?? undefined}>
                    <td>{r.mode}</td>
                    <td>{r.num_requests}</td>
                    <td>{r.wall_time_s.toFixed(2)}</td>
                    <td>{r.requests_per_sec.toFixed(2)}</td>
                    <td>{r.p50_latency_ms.toFixed(1)}</td>
                    <td>{r.p95_latency_ms.toFixed(1)}</td>
                    <td>{r.avg_ttft_ms.toFixed(1)}</td>
                    <td>{r.avg_tokens_per_sec.toFixed(1)}</td>
                    <td>{r.total_cost_usd.toFixed(5)}</td>
                    <td>{r.est_cost_per_1m_tokens.toFixed(4)}</td>
                    <td>
                      {r.cache_hit_rate != null
                        ? `${(r.cache_hit_rate * 100).toFixed(0)}%`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            Hover a row for the traffic shape used per mode. quantized is a
            simulated INT8/FP8 profile (TensorRT-LLM placeholder).
          </p>
        </>
      )}
    </div>
  );
}

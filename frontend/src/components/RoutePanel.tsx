import { useState } from "react";
import { api } from "../api";
import type { RouteResponse, RouterBenchmark } from "../types";
import BarChart from "./BarChart";

export default function RoutePanel() {
  const [prompt, setPrompt] = useState("I was charged twice this month.");
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [bench, setBench] = useState<RouterBenchmark | null>(null);
  const [busy, setBusy] = useState(false);
  const [benchBusy, setBenchBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const classify = async () => {
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      setRoute(await api.route(prompt.trim()));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const runBench = async () => {
    setBenchBusy(true);
    setError(null);
    try {
      setBench(await api.routerBenchmark(5000));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBenchBusy(false);
    }
  };

  return (
    <div className="card">
      <h2>Semantic router</h2>
      <p className="hint">
        CUDA-accelerated intent classification (CuPy on NVIDIA GPUs, NumPy
        fallback on CPU) that routes traffic to small vs large model tiers.
      </p>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        aria-label="Prompt to classify"
      />
      <div className="row">
        <button onClick={classify} disabled={busy || !prompt.trim()}>
          {busy ? "Classifying…" : "Classify intent"}
        </button>
        <button className="ghost" onClick={runBench} disabled={benchBusy}>
          {benchBusy ? "Benchmarking…" : "CPU vs GPU benchmark"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {route && (
        <div className="route-result">
          <div className="route-cat">{route.category}</div>
          <div className="route-meta">
            confidence {(route.confidence * 100).toFixed(1)}% · tier:{" "}
            {route.model_tier} model · {route.route_time_ms.toFixed(3)}ms ·{" "}
            {route.backend}
          </div>
          <BarChart
            title="Similarity scores"
            bars={Object.entries(route.scores)
              .sort((a, b) => b[1] - a[1])
              .map(([name, value]) => ({ name, value: Math.max(value, 0) }))}
            format={(v) => v.toFixed(3)}
          />
        </div>
      )}

      {bench && (
        <div className="route-result">
          <BarChart
            title={`Similarity scoring: ${bench.num_intents.toLocaleString()} intents × ${bench.embed_dim}-dim`}
            bars={[
              { name: "CPU (NumPy)", value: bench.cpu_time_ms },
              ...(bench.gpu_time_ms != null
                ? [{ name: "GPU (CuPy)", value: bench.gpu_time_ms }]
                : []),
            ]}
            format={(v) => `${v.toFixed(3)}ms`}
          />
          <p className="note">
            {bench.speedup != null
              ? `GPU speedup: ${bench.speedup.toFixed(2)}x — ${bench.note}`
              : bench.note}
          </p>
        </div>
      )}
    </div>
  );
}

import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { MetricsSnapshot } from "./types";
import BenchmarkPanel from "./components/BenchmarkPanel";
import ChatPanel from "./components/ChatPanel";
import MetricsCards from "./components/MetricsCards";
import RoutePanel from "./components/RoutePanel";

export default function App() {
  const [snap, setSnap] = useState<MetricsSnapshot | null>(null);
  const [offline, setOffline] = useState(false);

  const refresh = useCallback(() => {
    api
      .metrics()
      .then((s) => {
        setSnap(s);
        setOffline(false);
      })
      .catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <div className="app">
      <header className="header">
        <h1>
          Infer<span>Edge</span>
        </h1>
        <span className="subtitle">
          GPU-optimized LLM inference platform for real-time AI agents
        </span>
      </header>
      <div className="badges">
        <span className="badge">
          inference: <strong>{snap?.inference_mode ?? "…"}</strong>
        </span>
        <span className="badge">
          router:{" "}
          <strong>
            {snap?.router_benchmark?.backend ??
              (snap?.recent.find((r) => r.router_backend)?.router_backend ??
                "run a routed request")}
          </strong>
        </span>
        <span className="badge">
          requests served: <strong>{snap?.total_requests ?? 0}</strong>
        </span>
        {offline && (
          <span className="badge" style={{ color: "var(--bad)" }}>
            backend unreachable — start uvicorn on :8000
          </span>
        )}
      </div>

      <MetricsCards snap={snap} />

      <div className="grid" style={{ marginTop: 16 }}>
        <ChatPanel onActivity={refresh} />
        <RoutePanel />
        <BenchmarkPanel onActivity={refresh} />
      </div>

      <footer>
        InferEdge · FastAPI + React + CUDA (CuPy) · NVIDIA NIM / Nemotron
        ready · metrics window: last {snap?.window_size ?? 0} requests
      </footer>
    </div>
  );
}

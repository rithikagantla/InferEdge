import type {
  BenchmarkResult,
  ChatResponse,
  MetricsSnapshot,
  OptimizationMode,
  RouteResponse,
  RouterBenchmark,
} from "./types";

// Vite dev server and nginx both proxy /api/* to the FastAPI backend.
const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`API ${resp.status}: ${detail}`);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  chat: (prompt: string, mode: OptimizationMode) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ prompt, mode }),
    }),

  batchChat: (prompts: string[]) =>
    request<{ responses: ChatResponse[]; batch_size: number; wall_time_ms: number; requests_per_sec: number }>(
      "/batch-chat",
      { method: "POST", body: JSON.stringify({ prompts }) },
    ),

  metrics: () => request<MetricsSnapshot>("/metrics"),

  route: (prompt: string) =>
    request<RouteResponse>("/route", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    }),

  routerBenchmark: (numIntents = 5000) =>
    request<RouterBenchmark>("/router/benchmark", {
      method: "POST",
      body: JSON.stringify({ num_intents: numIntents }),
    }),

  runBenchmark: (numPrompts: number) =>
    request<BenchmarkResult>("/benchmark/run", {
      method: "POST",
      body: JSON.stringify({ num_prompts: numPrompts }),
    }),

  benchmarkResults: () =>
    request<{ runs: BenchmarkResult[] }>("/benchmark-results"),

  samplePrompts: () => request<{ prompts: string[] }>("/sample-prompts"),
};

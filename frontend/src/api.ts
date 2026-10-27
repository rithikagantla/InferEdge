import type {
  AppConfig,
  BenchmarkResult,
  ChatMode,
  ChatResponse,
  MeasuredResults,
  MetricsSnapshot,
  RouteResponse,
  RouterBenchmark,
} from "./types";

// Vite dev server and nginx both proxy /api/* to the FastAPI backend.
const BASE = "/api";
const UNREACHABLE = "Can't reach the InferEdge backend. Start it on port 8000 and retry.";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(UNREACHABLE, 0);
  }
  if (!resp.ok) {
    let body: { detail?: unknown } | null = null;
    try {
      body = await resp.json();
    } catch {
      /* non-JSON error body */
    }
    // FastAPI always answers with JSON; a bare 5xx comes from the dev/nginx
    // proxy when the backend itself isn't running.
    if (!body && resp.status >= 500) throw new ApiError(UNREACHABLE, resp.status);
    let detail = resp.statusText;
    if (typeof body?.detail === "string") detail = body.detail;
    else if (Array.isArray(body?.detail)) detail = body!.detail.map((d: { msg: string }) => d.msg).join("; ");
    throw new ApiError(detail || `Request failed (${resp.status})`, resp.status);
  }
  return resp.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  chat: (prompt: string, mode: ChatMode, history: { role: string; content: string }[]) =>
    post<ChatResponse>("/chat", { prompt, mode, history }),
  config: () => request<AppConfig>("/config"),
  health: () => request<{ status: string }>("/health"),
  metrics: () => request<MetricsSnapshot>("/metrics"),
  results: () => request<MeasuredResults>("/results"),
  route: (prompt: string) => post<RouteResponse>("/route", { prompt }),
  routerBenchmark: (numIntents = 5000) =>
    post<RouterBenchmark>("/router/benchmark", { num_intents: numIntents }),
  runBenchmark: (numPrompts: number) => post<BenchmarkResult>("/benchmark/run", { num_prompts: numPrompts }),
  benchmarkResults: () => request<{ runs: BenchmarkResult[] }>("/benchmark-results"),
  samplePrompts: () => request<{ prompts: string[] }>("/sample-prompts"),
};

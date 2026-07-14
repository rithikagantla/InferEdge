export type OptimizationMode =
  | "baseline"
  | "batching"
  | "caching"
  | "routing"
  | "quantized";

export const OPTIMIZATION_MODES: OptimizationMode[] = [
  "baseline",
  "batching",
  "caching",
  "routing",
  "quantized",
];

export interface RequestMetrics {
  mode: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  ttft_ms: number;
  latency_ms: number;
  tokens_per_sec: number;
  cost_usd: number;
  cached: boolean;
  category: string | null;
  router_backend: string | null;
  router_time_ms: number | null;
}

export interface ChatResponse {
  response: string;
  metrics: RequestMetrics;
}

export interface Aggregate {
  requests: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  avg_ttft_ms: number;
  avg_tokens_per_sec: number;
  requests_per_sec: number | null;
  input_tokens: number;
  output_tokens: number;
  total_cost_usd: number;
  est_cost_per_1m_tokens: number;
  cache_hit_rate: number;
}

export interface RouterBenchmark {
  num_intents: number;
  embed_dim: number;
  cpu_time_ms: number;
  gpu_time_ms: number | null;
  speedup: number | null;
  gpu_available: boolean;
  backend: string;
  note: string;
}

export interface MetricsSnapshot {
  inference_mode: string;
  total_requests: number;
  window_size: number;
  overall: Aggregate;
  by_mode: Record<string, Aggregate>;
  avg_router_time_ms: number | null;
  router_benchmark: RouterBenchmark | null;
  cache: { hits: number; misses: number; hit_rate: number };
  recent: RequestMetrics[];
}

export interface RouteResponse {
  category: string;
  confidence: number;
  scores: Record<string, number>;
  model_tier: string;
  backend: string;
  route_time_ms: number;
}

export interface ModeResult {
  mode: string;
  num_requests: number;
  wall_time_s: number;
  requests_per_sec: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  avg_ttft_ms: number;
  avg_tokens_per_sec: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  est_cost_per_1m_tokens: number;
  cache_hit_rate: number | null;
  notes: string | null;
}

export interface BenchmarkResult {
  id: number | null;
  created_at: string;
  inference_mode: string;
  num_prompts: number;
  results: ModeResult[];
}

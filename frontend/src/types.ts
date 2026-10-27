export type ChatMode = "routing" | "baseline" | "caching" | "quantized";
export type Tier = "small" | "large" | "quantized";

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
  tier: Tier | null;
  input_price_per_1m: number | null;
  output_price_per_1m: number | null;
  route_confidence: number | null;
  route_scores: Record<string, number> | null;
  history_turns: number;
}

export interface ChatResponse {
  response: string;
  metrics: RequestMetrics;
}

export interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: ChatMode;
  metrics?: RequestMetrics;
  error?: string;
  pending?: boolean;
  animate?: boolean;
  createdAt: number;
}

export interface TierInfo {
  model: string;
  input_cost_per_1m: number;
  output_cost_per_1m: number;
}

export interface AppConfig {
  inference_mode: string;
  router_backend: string;
  router_embeddings: string;
  small_tier_categories: string[];
  tiers: Record<string, TierInfo>;
  chat_modes: { id: ChatMode; description: string }[];
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

export interface MetricsSnapshot {
  inference_mode: string;
  total_requests: number;
  window_size: number;
  overall: Aggregate;
  by_mode: Record<string, Aggregate>;
  router_benchmark: RouterBenchmark | null;
  cache: { hits: number; misses: number; hit_rate: number };
}

export interface RouteResponse {
  category: string;
  confidence: number;
  scores: Record<string, number>;
  model_tier: string;
  backend: string;
  route_time_ms: number;
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

export interface ModeResult {
  mode: string;
  num_requests: number;
  wall_time_s: number;
  requests_per_sec: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  avg_ttft_ms: number;
  avg_tokens_per_sec: number;
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

export interface SweepLevel {
  concurrency: number;
  requests: number;
  errors: number;
  output_tokens_per_sec: number;
  requests_per_sec: number;
  ttft_p50_ms: number;
  ttft_p95_ms: number;
  tpot_p50_ms: number;
  tpot_p95_ms: number;
  itl_p99_ms: number;
  e2e_p50_ms: number;
  avg_output_tokens: number;
}

export interface Sweep {
  label: string;
  engine: string | null;
  engine_version: string | null;
  precision: string | null;
  variant: string | null;
  gpu_name: string | null;
  gpu_memory_peak_used_mib: number | null;
  ignore_eos: boolean | null;
  model: string | null;
  levels: SweepLevel[];
}

export interface RoutingStudyRow {
  strategy: string;
  weak_share: number;
  quality_mean: number;
  quality_hard: number;
  good_or_better: number;
  ttft_p50_ms: number;
  ttft_p95_ms: number;
  total_cost_usd: number;
  judge_tokens: number;
  judge_p50_ms: number | null;
  quality_vs_ultra?: { mean: number; ci95: [number, number] };
}

export interface MeasuredResults {
  nim_benchmark: BenchmarkResult | null;
  nim_sweeps: Sweep[];
  gpu_sweeps: Sweep[];
  router_gpu: {
    gpu: string;
    rows: { intents: number; cpu_ms: number; gpu_compute_ms: number; gpu_end_to_end_ms: number }[];
  } | null;
  router_embed: {
    gpu: string;
    cpu_cores: number;
    eval_set_size: number;
    accuracy: Record<string, { category_accuracy: number; tier_accuracy: number }>;
    latency: { messages: number; cpu_ms: number; gpu_resident_ms: number; gpu_messages_per_s: number }[];
  } | null;
  routing_study: RoutingStudyRow[] | null;
}

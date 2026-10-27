/** Fixed categorical order (never cycled by rank): a series keeps its color
 *  when others are toggled off. Values are CSS variables so dark mode uses
 *  its own lighter steps. */
export const SERIES_COLORS = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
  "var(--series-5)",
  "var(--series-6)",
];

export const SERIES_MARKERS = ["circle", "square", "triangle", "diamond", "wye", "cross"] as const;

const LABELS: Record<string, string> = {
  "trtllm-fp8-L4": "TensorRT-LLM · FP8",
  "vllm-fp8-kv-L4": "vLLM · FP8 + FP8 KV cache",
  "vllm-fp8-L4": "vLLM · FP8",
  "trtllm-bf16-L4": "TensorRT-LLM · bf16",
  "vllm-bf16-L4": "vLLM · bf16",
  "trtllm-pytorch-bf16-L4": "TensorRT-LLM PyTorch · bf16",
  "trtllm-pytorch-bf16-natural-L4": "No speculation",
  "trtllm-spec-draft-L4": "Draft model (Llama 3.2 1B)",
  "trtllm-spec-ngram-private-L4": "N-gram (private pool)",
  "trtllm-spec-ngram-L4": "N-gram (shared pool, cache artifact)",
  "trtllm-fp8-bs128-L4": "Batch 128 · 8k tokens/step",
  "trtllm-fp8-bs256-L4": "Batch 256 · 8k tokens/step",
  "trtllm-fp8-bs256-tok16k-L4": "Batch 256 · 16k tokens/step",
  "nim-nemotron3-ultra": "Nemotron 3 Ultra",
  "nim-nemotron3.5-lightning": "Nemotron 3.5 Lightning",
};

export function seriesName(label: string): string {
  return LABELS[label] ?? label;
}

export const SWEEP_GROUPS: { id: string; name: string; labels: string[]; note: string }[] = [
  {
    id: "engines",
    name: "Engines & precision",
    labels: ["trtllm-fp8-L4", "vllm-fp8-kv-L4", "vllm-fp8-L4", "trtllm-bf16-L4", "vllm-bf16-L4", "trtllm-pytorch-bf16-L4"],
    note: "Llama 3 8B Instruct, 1–64 concurrent requests, exactly 256 output tokens each.",
  },
  {
    id: "speculative",
    name: "Speculative decoding",
    labels: ["trtllm-spec-draft-L4", "trtllm-pytorch-bf16-natural-L4", "trtllm-spec-ngram-private-L4"],
    note: "TensorRT-LLM PyTorch backend, bf16, natural-length answers (identical output with greedy decoding).",
  },
  {
    id: "batching",
    name: "Batching limits",
    labels: ["trtllm-fp8-bs256-tok16k-L4", "trtllm-fp8-bs256-L4", "trtllm-fp8-bs128-L4", "trtllm-fp8-L4"],
    note: "The FP8 engine rebuilt for higher concurrency, swept to 256 concurrent requests.",
  },
];

export const STRATEGY_NAMES: Record<string, string> = {
  "always-ultra": "Always Ultra",
  "always-lightning": "Always Lightning",
  "inferedge-semantic": "InferEdge router",
  "sy-classifier": "Switchyard LLM classifier",
  "sy-random": "Switchyard random 50/50",
  "sy-random-matched": "Switchyard random, budget-matched",
};

/** SAMPLE_PROMPTS is ordered in blocks of five per category. */
export const SAMPLE_CATEGORIES = [
  "shipping", "billing", "account_management", "refund", "technical_support", "human_escalation",
];

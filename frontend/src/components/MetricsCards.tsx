import type { MetricsSnapshot } from "../types";

function fmtMs(v: number): string {
  return v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${v.toFixed(1)}ms`;
}

export default function MetricsCards({ snap }: { snap: MetricsSnapshot | null }) {
  const o = snap?.overall;
  const cards = [
    { label: "p50 latency", value: o && o.requests ? fmtMs(o.p50_latency_ms) : "—" },
    { label: "p95 latency", value: o && o.requests ? fmtMs(o.p95_latency_ms) : "—" },
    { label: "avg tokens/sec", value: o && o.requests ? o.avg_tokens_per_sec.toFixed(1) : "—" },
    {
      label: "requests/sec",
      value: o?.requests_per_sec != null ? o.requests_per_sec.toFixed(2) : "—",
    },
    {
      label: "est. cost / 1M tokens",
      value: o && o.requests ? `$${o.est_cost_per_1m_tokens.toFixed(3)}` : "—",
    },
    {
      label: "cache hit rate",
      value: snap ? `${(snap.cache.hit_rate * 100).toFixed(0)}%` : "—",
    },
  ];
  return (
    <div className="stats">
      {cards.map((c) => (
        <div className="stat" key={c.label}>
          <div className="label">{c.label}</div>
          <div className="value">{c.value}</div>
        </div>
      ))}
    </div>
  );
}

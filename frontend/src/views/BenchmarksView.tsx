import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Icon } from "../components/Icon";
import { Card, ErrorNote, Segmented, Stat } from "../components/ui";
import { useAsync } from "../hooks";
import { ms, num, pct, usd } from "../lib/format";
import type { ThemeColors } from "../lib/useThemeColors";
import type { BenchmarkResult, MetricsSnapshot, ModeResult } from "../types";

type MetricKey = "p50_latency_ms" | "p95_latency_ms" | "requests_per_sec" | "est_cost_per_1m_tokens";

const METRICS: { id: MetricKey; label: string; better: "lower" | "higher"; fmt: (v: number) => string }[] = [
  { id: "p50_latency_ms", label: "p50 latency", better: "lower", fmt: (v) => ms(v) },
  { id: "p95_latency_ms", label: "p95 latency", better: "lower", fmt: (v) => ms(v) },
  { id: "requests_per_sec", label: "Throughput", better: "higher", fmt: (v) => `${num(v, 2)} req/s` },
  { id: "est_cost_per_1m_tokens", label: "Cost / 1M tokens", better: "lower", fmt: (v) => `$${v.toFixed(3)}` },
];

const MODE_LABEL: Record<string, string> = {
  baseline: "Baseline", batching: "Batching", caching: "Caching", routing: "Routing", quantized: "FP8 (sim)",
};

export function BenchmarksView({ snapshot, colors, onActivity }: {
  snapshot: MetricsSnapshot | null;
  colors: ThemeColors;
  onActivity: () => void;
}) {
  const runs = useAsync(() => api.benchmarkResults());
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [metric, setMetric] = useState<MetricKey>("p50_latency_ms");
  const [numPrompts, setNumPrompts] = useState(20);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const allRuns = runs.data?.runs ?? [];
  const run: BenchmarkResult | undefined = allRuns.find((r) => r.id === selectedId) ?? allRuns[0];
  const m = METRICS.find((x) => x.id === metric)!;

  const data = useMemo(() => (run?.results ?? []).map((r) => ({ ...r, name: MODE_LABEL[r.mode] ?? r.mode })), [run]);
  const best = data.length
    ? data.reduce((b, r) => ((m.better === "lower" ? r[metric] < b[metric] : r[metric] > b[metric]) ? r : b), data[0])
    : null;

  const start = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.runBenchmark(numPrompts);
      runs.setData({ runs: [res, ...allRuns] });
      setSelectedId(res.id);
      onActivity();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const o = snapshot?.overall;
  return (
    <>
      <header className="page-header">
        <div>
          <h1 className="page-title">Optimization benchmark</h1>
          <p className="page-sub">
            Run the support-prompt suite through every serving strategy against the current backend
            ({snapshot?.inference_mode ?? "…"} mode) and compare latency, throughput and cost.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <label className="sr-only" htmlFor="num-prompts">Prompts per mode</label>
          <select id="num-prompts" className="btn" value={numPrompts} onChange={(e) => setNumPrompts(Number(e.target.value))}>
            {[10, 20, 50, 100].map((n) => <option key={n} value={n}>{n} prompts</option>)}
          </select>
          <button className="btn btn-primary" type="button" onClick={start} disabled={running}>
            {running ? <span className="spinner" aria-hidden /> : <Icon name="bench" size={15} />}
            {running ? "Running…" : "Run benchmark"}
          </button>
        </div>
      </header>
      <div className="page-body">
        {error && <ErrorNote message={error} onRetry={start} />}

        <div className="stats" aria-label="Live metrics, last 200 requests">
          <Stat label="p50 latency" value={o?.requests ? ms(o.p50_latency_ms) : "—"} foot="last 200 requests" />
          <Stat label="p95 latency" value={o?.requests ? ms(o.p95_latency_ms) : "—"} />
          <Stat label="Tokens / sec" value={o?.requests ? num(o.avg_tokens_per_sec, 1) : "—"} foot="per request, generated only" />
          <Stat label="Requests / sec" value={o?.requests_per_sec != null ? num(o.requests_per_sec, 2) : "—"} />
          <Stat label="Cost / 1M tokens" value={o?.requests ? `$${o.est_cost_per_1m_tokens.toFixed(3)}` : "—"} />
          <Stat label="Cache hit rate" value={snapshot ? pct(snapshot.cache.hit_rate) : "—"} foot={`${snapshot?.total_requests ?? 0} requests served`} />
        </div>

        <Card
          title="Compare strategies"
          sub={run ? `${run.num_prompts} prompts per mode · ${run.inference_mode} mode · ${new Date(run.created_at).toLocaleString()}` : "No runs yet: start one above"}
          actions={
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {allRuns.length > 1 && (
                <select className="btn btn-sm" value={run?.id ?? ""} aria-label="Benchmark run"
                  onChange={(e) => setSelectedId(Number(e.target.value))}>
                  {allRuns.map((r) => (
                    <option key={r.id ?? r.created_at} value={r.id ?? ""}>
                      Run #{r.id} · {r.num_prompts} prompts · {r.inference_mode}
                    </option>
                  ))}
                </select>
              )}
              <Segmented label="Metric" value={metric} onChange={setMetric}
                options={METRICS.map((x) => ({ id: x.id, label: x.label }))} />
            </div>
          }
        >
          {runs.error && <ErrorNote message={runs.error} onRetry={runs.reload} />}
          {data.length > 0 && (
            <>
              <p className="card-sub" style={{ marginTop: -6, marginBottom: 8 }}>
                {m.label}: {m.better} is better. Best: <b>{best?.name}</b> at {best && m.fmt(best[metric])}.
              </p>
              <div style={{ width: "100%", height: 280 }}>
                <ResponsiveContainer>
                  <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 8 }}>
                    <CartesianGrid vertical={false} stroke={colors.grid} />
                    <XAxis dataKey="name" tick={{ fill: colors.ink2, fontSize: 12.5 }} axisLine={{ stroke: colors.border }} tickLine={false} />
                    <YAxis tick={{ fill: colors.ink3, fontSize: 12 }} axisLine={false} tickLine={false}
                      tickFormatter={(v: number) => (metric === "est_cost_per_1m_tokens" ? `$${v.toFixed(2)}` : metric === "requests_per_sec" ? num(v, 1) : ms(v))}
                      width={64} />
                    <Tooltip cursor={{ fill: colors.grid, opacity: 0.6 }} content={<ModeTooltip />} />
                    <Bar dataKey={metric} radius={[4, 4, 0, 0]} maxBarSize={64} isAnimationActive>
                      {data.map((d) => (
                        <Cell key={d.mode} fill={d === best ? colors.accentFill : colors.ink3} fillOpacity={d === best ? 1 : 0.55} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="table-wrap" style={{ marginTop: 14 }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th>Mode</th><th>Req/s</th><th>p50</th><th>p95</th><th>TTFT</th><th>Tok/s</th><th>Total cost</th><th>$/1M tok</th><th>Cache</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.map((r) => (
                      <tr key={r.mode} className={r === best ? "highlight" : ""} title={r.notes ?? undefined}>
                        <td>{r.name}</td>
                        <td>{num(r.requests_per_sec, 2)}</td>
                        <td>{ms(r.p50_latency_ms)}</td>
                        <td>{ms(r.p95_latency_ms)}</td>
                        <td>{ms(r.avg_ttft_ms)}</td>
                        <td>{num(r.avg_tokens_per_sec, 1)}</td>
                        <td>{usd(r.total_cost_usd)}</td>
                        <td>${r.est_cost_per_1m_tokens.toFixed(3)}</td>
                        <td>{r.cache_hit_rate != null ? pct(r.cache_hit_rate) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="card-sub" style={{ marginTop: 10 }}>
                Hover a row for how each mode's traffic is generated. In mock mode the timings come from a
                simulated serving profile; see Results for measured numbers.
              </p>
            </>
          )}
        </Card>
      </div>
    </>
  );
}

function ModeTooltip({ active, payload }: { active?: boolean; payload?: { payload: ModeResult & { name: string } }[] }) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div className="tt-title">{r.name}</div>
      <div className="tt-row"><span>p50 / p95</span><b>{ms(r.p50_latency_ms)} / {ms(r.p95_latency_ms)}</b></div>
      <div className="tt-row"><span>Throughput</span><b>{num(r.requests_per_sec, 2)} req/s</b></div>
      <div className="tt-row"><span>Cost / 1M tokens</span><b>${r.est_cost_per_1m_tokens.toFixed(3)}</b></div>
      {r.cache_hit_rate != null && <div className="tt-row"><span>Cache hits</span><b>{pct(r.cache_hit_rate)}</b></div>}
    </div>
  );
}

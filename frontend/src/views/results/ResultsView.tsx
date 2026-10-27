import { useState } from "react";
import { api } from "../../api";
import { Bars, Card, ErrorNote, Segmented, Stat } from "../../components/ui";
import { useAsync } from "../../hooks";
import { ms, num, pct, usd } from "../../lib/format";
import { STRATEGY_NAMES, SWEEP_GROUPS, seriesName } from "../../lib/series";
import type { ThemeColors } from "../../lib/useThemeColors";
import type { MeasuredResults, Sweep } from "../../types";
import { SweepChart } from "./SweepChart";

type Tab = "gpu" | "nim" | "routing" | "router";

const GROUP_FINDINGS: Record<string, { title: string; body: string }[]> = {
  engines: [
    { title: "FP8 is the biggest lever", body: "Halving the bytes read per token makes one user's text stream ~1.6x faster and lifts peak throughput 56–63%." },
    { title: "TensorRT-LLM wins under load", body: "At 64 users: 131 ms vs 283–298 ms first-token wait and 48 vs 139–150 ms worst token gaps. vLLM is 2–3 ms/token faster for a single user." },
    { title: "Not the cache format", body: "vLLM with an FP8 KV cache doubled its cache (81k → 156k tokens) but barely moved latency, so TensorRT-LLM's edge is its scheduler and kernels." },
  ],
  speculative: [
    { title: "Draft model: 1.4x faster per user", body: "Llama 3.2 1B drafts, Llama 3 8B verifies; identical output with greedy decoding (59 → 41 ms per token)." },
    { title: "But it costs capacity", body: "Two models on 24 GB leave a third of the KV cache, so past ~8 users requests queue: 21 s first-token waits at 64." },
    { title: "N-gram didn't help", body: "Support answers rarely repeat their question, so drafts are rejected: 0.96x at one user, 0.47x at 64." },
  ],
  batching: [
    { title: "2x peak throughput", body: "Raising the engine's batch limit from 64 to 256 reached 2,928 tok/s with no cost below 64 users." },
    { title: "The knee is ~128 users", body: "2,331 tok/s with first-token p95 under 0.4 s; at 256 users each stream slows to ~12 tokens/s." },
    { title: "Leave headroom", body: "At 128 users, an engine capped at exactly 128 gave 1,969 tok/s; one capped at 256 gave 2,266." },
  ],
};

export function ResultsView({ colors }: { colors: ThemeColors }) {
  const { data, error, loading, reload } = useAsync(() => api.results());
  const [tab, setTab] = useState<Tab>("gpu");

  return (
    <>
      <header className="page-header">
        <div>
          <h1 className="page-title">Measured results</h1>
          <p className="page-sub">
            Real measurements, not the simulator: hosted NVIDIA NIM, Llama 3 8B on a cloud NVIDIA L4 with
            vLLM and TensorRT-LLM, the semantic router on GPU, and a routing study against NVIDIA NeMo Switchyard.
          </p>
        </div>
        <Segmented<Tab> label="Result set" value={tab} onChange={setTab} options={[
          { id: "gpu", label: "Self-hosted L4" },
          { id: "nim", label: "Hosted NIM" },
          { id: "routing", label: "Routing study" },
          { id: "router", label: "Router" },
        ]} />
      </header>
      <div className="page-body">
        {error && <ErrorNote message={error} onRetry={reload} />}
        {loading && !data && <p className="muted">Loading results…</p>}
        {data && tab === "gpu" && <GpuTab data={data} colors={colors} />}
        {data && tab === "nim" && <NimTab data={data} colors={colors} />}
        {data && tab === "routing" && <RoutingTab data={data} />}
        {data && tab === "router" && <RouterTab data={data} />}
      </div>
    </>
  );
}

function pick(sweeps: Sweep[], labels: string[]): Sweep[] {
  return labels.map((l) => sweeps.find((s) => s.label === l)).filter((s): s is Sweep => !!s);
}

function GpuTab({ data, colors }: { data: MeasuredResults; colors: ThemeColors }) {
  const [group, setGroup] = useState(SWEEP_GROUPS[0].id);
  const g = SWEEP_GROUPS.find((x) => x.id === group)!;
  const sweeps = pick(data.gpu_sweeps, g.labels);
  if (!data.gpu_sweeps.length) return <Card title="No GPU results"><p className="muted">Run the Modal scripts in gpu/ to produce them.</p></Card>;
  const top = (s: Sweep) => { const ok = s.levels.filter((l) => l.requests > 0); return ok[ok.length - 1]; };

  return (
    <>
      <Card
        title="Throughput vs latency on one NVIDIA L4"
        sub={g.note}
        actions={<Segmented label="Experiment" value={group} onChange={setGroup}
          options={SWEEP_GROUPS.map((x) => ({ id: x.id, label: x.name }))} />}
      >
        <SweepChart key={group} sweeps={sweeps} colors={colors} />
      </Card>
      <div className="findings">
        {GROUP_FINDINGS[group].map((f) => (
          <div className="finding" key={f.title}><strong>{f.title}</strong>{f.body}</div>
        ))}
      </div>
      <Card title="At a glance" sub="One user vs the highest concurrency each configuration was swept to">
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr><th>Configuration</th><th>1 user: ms/token</th><th>1 user: TTFT</th><th>Top level</th><th>Throughput</th><th>TTFT p50 / p95</th><th>Worst gap</th><th>Peak GPU mem</th></tr>
            </thead>
            <tbody>
              {sweeps.map((s) => {
                const one = s.levels[0];
                const t = top(s);
                return (
                  <tr key={s.label}>
                    <td>{seriesName(s.label)}</td>
                    <td>{ms(one.tpot_p50_ms, 1)}</td>
                    <td>{ms(one.ttft_p50_ms)}</td>
                    <td>{t.concurrency} users</td>
                    <td>{num(t.output_tokens_per_sec)} tok/s</td>
                    <td>{ms(t.ttft_p50_ms)} / {ms(t.ttft_p95_ms)}</td>
                    <td>{ms(t.itl_p99_ms)}</td>
                    <td>{s.gpu_memory_peak_used_mib ? `${(s.gpu_memory_peak_used_mib / 1024).toFixed(1)} GB` : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}

function NimTab({ data, colors }: { data: MeasuredResults; colors: ThemeColors }) {
  const b = data.nim_benchmark;
  const sweeps = pick(data.nim_sweeps, ["nim-nemotron3.5-lightning", "nim-nemotron3-ultra"]);
  const base = b?.results.find((r) => r.mode === "baseline");
  const routing = b?.results.find((r) => r.mode === "routing");
  return (
    <>
      {base && routing && (
        <div className="stats">
          <Stat label="Routing: faster median" value={`${num(base.p50_latency_ms / routing.p50_latency_ms, 1)}×`} foot={`${ms(routing.p50_latency_ms)} vs ${ms(base.p50_latency_ms)}`} />
          <Stat label="Routing: cheaper per token" value={pct(1 - routing.est_cost_per_1m_tokens / base.est_cost_per_1m_tokens)} />
          <Stat label="Baseline p95 / p50" value={`${num(base.p95_latency_ms / base.p50_latency_ms, 1)}×`} foot="free-tier tail latency" />
          <Stat label="Cache hit rate" value={pct(b?.results.find((r) => r.mode === "caching")?.cache_hit_rate ?? null)} />
        </div>
      )}
      <Card title="50-prompt benchmark on hosted NIM" sub="Nemotron 3 Ultra (large) and Nemotron 3.5 Lightning (small), thinking disabled">
        {b ? (
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Mode</th><th>Req/s</th><th>p50</th><th>p95</th><th>Avg TTFT</th><th>Tok/s</th><th>$/1M tokens</th><th>Cache</th></tr></thead>
              <tbody>
                {b.results.map((r) => (
                  <tr key={r.mode}>
                    <td>{r.mode}</td><td>{num(r.requests_per_sec, 2)}</td><td>{ms(r.p50_latency_ms)}</td>
                    <td>{ms(r.p95_latency_ms)}</td><td>{ms(r.avg_ttft_ms)}</td><td>{num(r.avg_tokens_per_sec)}</td>
                    <td>${r.est_cost_per_1m_tokens.toFixed(3)}</td><td>{r.cache_hit_rate != null ? pct(r.cache_hit_rate) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="muted">No hosted NIM run found.</p>}
      </Card>
      {sweeps.length > 0 && (
        <Card title="Concurrency sweep on the free tier" sub="1–8 concurrent requests, natural-length answers">
          <SweepChart sweeps={sweeps} colors={colors} height={320} />
        </Card>
      )}
    </>
  );
}

function RoutingTab({ data }: { data: MeasuredResults }) {
  const rows = data.routing_study ?? [];
  if (!rows.length) return <Card title="No routing study found"><p className="muted">See docs/SWITCHYARD.md.</p></Card>;
  const ultra = rows.find((r) => r.strategy === "always-ultra");
  return (
    <>
      <div className="findings">
        <div className="finding"><strong>Quality was a statistical tie</strong>All paired 95% intervals vs always-Ultra include zero: even the small model alone matched it on short support turns.</div>
        <div className="finding"><strong>The LLM judge cost more than it saved</strong>Each Switchyard verdict read ~1,125 tokens, 8x an average answer: +51% cost and ~3 s more before the first token.</div>
        <div className="finding"><strong>InferEdge's router: 65% cheaper</strong>No routing delay, with the large model kept for technical issues and escalations.</div>
      </div>
      <div className="grid-2">
        <Card title="Cost, including the routing judge" sub="Same 50 prompts; assumed per-token prices">
          <Bars rows={rows.map((r) => ({ label: STRATEGY_NAMES[r.strategy] ?? r.strategy, value: r.total_cost_usd }))}
            format={(v) => usd(v)} highlightMax={false} />
        </Card>
        <Card title="Median wait for the first token" sub="Includes the judge call for the classifier">
          <Bars rows={rows.map((r) => ({ label: STRATEGY_NAMES[r.strategy] ?? r.strategy, value: r.ttft_p50_ms }))}
            format={(v) => ms(v)} highlightMax={false} />
        </Card>
      </div>
      <Card title="Blind-graded quality and routing" sub="gpt-oss-20b graded every answer 1–5 without knowing which model wrote it">
        <div className="table-wrap">
          <table className="data">
            <thead><tr><th>Strategy</th><th>Small-model share</th><th>Quality</th><th>vs always-Ultra (95% CI)</th><th>Hard requests</th><th>TTFT p50</th><th>Cost</th><th>vs Ultra</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.strategy}>
                  <td>{STRATEGY_NAMES[r.strategy] ?? r.strategy}</td>
                  <td>{pct(r.weak_share)}</td>
                  <td>{r.quality_mean.toFixed(2)}</td>
                  <td>{r.quality_vs_ultra ? `${r.quality_vs_ultra.mean >= 0 ? "+" : ""}${r.quality_vs_ultra.mean.toFixed(2)} [${r.quality_vs_ultra.ci95[0].toFixed(2)}, ${r.quality_vs_ultra.ci95[1].toFixed(2)}]` : "—"}</td>
                  <td>{r.quality_hard.toFixed(1)}</td>
                  <td>{ms(r.ttft_p50_ms)}</td>
                  <td>{usd(r.total_cost_usd)}</td>
                  <td>{ultra ? `${r.total_cost_usd >= ultra.total_cost_usd ? "+" : ""}${Math.round((r.total_cost_usd / ultra.total_cost_usd - 1) * 100)}%` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}

function RouterTab({ data }: { data: MeasuredResults }) {
  const e = data.router_embed;
  const g = data.router_gpu;
  return (
    <>
      {e && (
        <div className="grid-2">
          <Card title="Routing accuracy on held-out messages" sub={`${e.eval_set_size} labeled support messages the router never saw`}>
            <Bars rows={[
              { label: "Hashed words: category", value: e.accuracy.hashed.category_accuracy },
              { label: "MiniLM: category", value: e.accuracy.minilm.category_accuracy },
              { label: "Hashed words: tier", value: e.accuracy.hashed.tier_accuracy },
              { label: "MiniLM: tier", value: e.accuracy.minilm.tier_accuracy },
            ]} format={(v) => pct(v)} />
          </Card>
          <Card title="Text in, category out" sub={`MiniLM on ${e.gpu} vs ${e.cpu_cores} CPU cores`}>
            <div className="table-wrap">
              <table className="data">
                <thead><tr><th>Messages</th><th>CPU</th><th>GPU</th><th>Speedup</th><th>GPU msgs/s</th></tr></thead>
                <tbody>
                  {e.latency.map((r) => (
                    <tr key={r.messages}>
                      <td>{num(r.messages)}</td><td>{ms(r.cpu_ms, 1)}</td><td>{ms(r.gpu_resident_ms, 1)}</td>
                      <td>{num(r.cpu_ms / r.gpu_resident_ms, 1)}×</td><td>{num(r.gpu_messages_per_s)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
      {g && (
        <Card title="Scoring alone: CuPy vs NumPy" sub={`${g.gpu} · hashed embeddings, 384 dimensions × 6 intents`}>
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Intents</th><th>CPU</th><th>GPU compute</th><th>GPU incl. transfers</th><th>Speedup, compute</th><th>Speedup, end to end</th></tr></thead>
              <tbody>
                {g.rows.map((r) => (
                  <tr key={r.intents}>
                    <td>{num(r.intents)}</td><td>{ms(r.cpu_ms, 3)}</td><td>{ms(r.gpu_compute_ms, 3)}</td><td>{ms(r.gpu_end_to_end_ms, 3)}</td>
                    <td>{num(r.cpu_ms / r.gpu_compute_ms, 2)}×</td><td>{num(r.cpu_ms / r.gpu_end_to_end_ms, 2)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="card-sub" style={{ marginTop: 10 }}>
            Six dot products per intent are too little work to justify the trip over PCIe; a transformer forward pass (above) is not.
          </p>
        </Card>
      )}
    </>
  );
}

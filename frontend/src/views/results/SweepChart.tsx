import { useMemo, useState } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { Segmented } from "../../components/ui";
import { ms, num } from "../../lib/format";
import { SERIES_MARKERS, seriesName } from "../../lib/series";
import type { ThemeColors } from "../../lib/useThemeColors";
import type { Sweep, SweepLevel } from "../../types";

type XKey = "tpot_p50_ms" | "ttft_p50_ms";

/** Round tick values (1-2-5 steps) inside [lo, hi]. */
function niceTicks(lo: number, hi: number, log: boolean): number[] {
  if (log) {
    const out: number[] = [];
    for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) {
      for (const m of [1, 2, 5]) {
        const v = m * 10 ** p;
        if (v >= lo && v <= hi) out.push(v);
      }
    }
    return out;
  }
  const raw = (hi - lo) / 6;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

interface Point extends SweepLevel {
  series: string;
  x: number;
  y: number;
}

/** Throughput vs latency, one line per sweep, one point per concurrency
 *  level. Color follows the series (fixed order), never its rank. */
export function SweepChart({ sweeps, colors, height = 360 }: { sweeps: Sweep[]; colors: ThemeColors; height?: number }) {
  const [xKey, setXKey] = useState<XKey>("tpot_p50_ms");
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const visible = sweeps.filter((s) => !hidden.has(s.label));
  const allX = visible.flatMap((s) => s.levels.map((l) => l[xKey])).filter((v) => v > 0);
  const log = allX.length > 0 && Math.max(...allX) / Math.min(...allX) > 20;
  const domainX: [number, number] = allX.length
    ? [log ? Math.min(...allX) * 0.8 : Math.min(...allX) * 0.9, Math.max(...allX) * (log ? 1.5 : 1.08)]
    : [0, 1];

  const series = useMemo(() => sweeps.map((s, i) => ({
    sweep: s,
    color: colors.series[i % colors.series.length],
    marker: SERIES_MARKERS[i % SERIES_MARKERS.length],
    points: s.levels
      .filter((l) => l.requests > 0)
      .map((l) => ({ ...l, series: s.label, x: l[xKey], y: l.output_tokens_per_sec })) as Point[],
  })), [sweeps, colors, xKey]);

  const toggle = (label: string) => setHidden((h) => {
    const next = new Set(h);
    if (next.has(label)) next.delete(label); else if (next.size < sweeps.length - 1) next.add(label);
    return next;
  });

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }} role="group" aria-label="Series">
          {series.map((s) => (
            <button key={s.sweep.label} type="button" className="chip" aria-pressed={!hidden.has(s.sweep.label)}
              onClick={() => toggle(s.sweep.label)}>
              <span className="swatch" style={{ background: s.color }} />
              {seriesName(s.sweep.label)}
            </button>
          ))}
        </div>
        <Segmented label="X axis" value={xKey} onChange={setXKey}
          options={[{ id: "tpot_p50_ms", label: "Time per token" }, { id: "ttft_p50_ms", label: "Time to first token" }]} />
      </div>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 10, right: 16, bottom: 26, left: 8 }}>
            <CartesianGrid stroke={colors.grid} />
            <XAxis
              type="number" dataKey="x" scale={log ? "log" : "linear"} domain={domainX} allowDataOverflow
              ticks={niceTicks(domainX[0], domainX[1], log)} interval={0}
              tick={{ fill: colors.ink3, fontSize: 12 }} axisLine={{ stroke: colors.border }} tickLine={false}
              tickFormatter={(v: number) => ms(v)}
              label={{ value: `Median ${xKey === "tpot_p50_ms" ? "time per output token" : "time to first token"}${log ? " (log scale)" : ""} → slower`, position: "bottom", offset: 8, fill: colors.ink2, fontSize: 12.5 }}
            />
            <YAxis
              type="number" dataKey="y" tick={{ fill: colors.ink3, fontSize: 12 }} axisLine={false} tickLine={false} width={70}
              tickFormatter={(v: number) => num(v)}
              label={{ value: "Output tokens/s (all requests)", angle: -90, position: "insideLeft", offset: 4, fill: colors.ink2, fontSize: 12.5, style: { textAnchor: "middle" } }}
            />
            <ZAxis range={[70, 70]} />
            <Tooltip cursor={{ stroke: colors.border, strokeDasharray: "3 3" }} content={<SweepTooltip colorOf={(l) => series.find((s) => s.sweep.label === l)?.color} />} />
            {series.filter((s) => !hidden.has(s.sweep.label)).map((s) => (
              <Scatter key={s.sweep.label} name={seriesName(s.sweep.label)} data={s.points}
                fill={s.color} line={{ stroke: s.color, strokeWidth: 2 }} lineType="joint" shape={s.marker}
                isAnimationActive={false} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="card-sub" style={{ marginTop: 2 }}>
        Each point is a concurrency level; hover for details. Up and to the left is better. Click a series to hide it.
      </p>
    </div>
  );
}

function SweepTooltip({ active, payload, colorOf }: {
  active?: boolean;
  payload?: { payload: Point }[];
  colorOf: (label: string) => string | undefined;
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div className="tt-title"><span className="swatch" style={{ background: colorOf(p.series) }} />{seriesName(p.series)}</div>
      <div className="tt-row"><span>Concurrent requests</span><b>{p.concurrency}</b></div>
      <div className="tt-row"><span>Output throughput</span><b>{num(p.output_tokens_per_sec)} tok/s</b></div>
      <div className="tt-row"><span>Time per token (p50)</span><b>{ms(p.tpot_p50_ms, 1)}</b></div>
      <div className="tt-row"><span>First token p50 / p95</span><b>{ms(p.ttft_p50_ms)} / {ms(p.ttft_p95_ms)}</b></div>
      <div className="tt-row"><span>Worst token gap (p99)</span><b>{ms(p.itl_p99_ms)}</b></div>
      {p.errors > 0 && <div className="tt-row"><span>Errors</span><b>{p.errors}</b></div>}
    </div>
  );
}

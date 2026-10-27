import type { AppConfig, ChatMode, ChatTurn } from "../../types";
import { ms, pct, usd } from "../../lib/format";

const MODE_NAMES: Record<string, string> = {
  routing: "Smart routing",
  baseline: "Large model only",
  caching: "Response cache",
  quantized: "Simulated FP8",
};

export function SessionPanel({ turns, config, mode, onMode }: {
  turns: ChatTurn[];
  config: AppConfig | null;
  mode: ChatMode;
  onMode: (m: ChatMode) => void;
}) {
  const answered = turns.filter((t) => t.role === "assistant" && t.metrics && !t.error);
  const large = config?.tiers.large;
  const cost = answered.reduce((s, t) => s + t.metrics!.cost_usd, 0);
  const largeCost = large
    ? answered.reduce((s, t) => s + (t.metrics!.input_tokens * large.input_cost_per_1m
        + t.metrics!.output_tokens * large.output_cost_per_1m) / 1e6, 0)
    : 0;
  const ttfts = answered.map((t) => t.metrics!.ttft_ms).sort((a, b) => a - b);
  const medianTtft = ttfts.length ? ttfts[Math.floor((ttfts.length - 1) / 2)] : null;
  const counts = {
    small: answered.filter((t) => t.metrics!.tier === "small" && !t.metrics!.cached).length,
    large: answered.filter((t) => t.metrics!.tier !== "small" && !t.metrics!.cached).length,
    cache: answered.filter((t) => t.metrics!.cached).length,
  };
  const total = answered.length || 1;

  return (
    <aside className="session" aria-label="Session">
      <div className="card">
        <h3>Answer with</h3>
        <div className="mode-list" role="radiogroup" aria-label="Optimization mode">
          {(config?.chat_modes ?? [{ id: "routing" as ChatMode, description: "" }]).map((o) => (
            <div
              key={o.id}
              className="mode-option"
              role="radio"
              tabIndex={0}
              aria-checked={o.id === mode}
              onClick={() => onMode(o.id)}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onMode(o.id))}
            >
              <span className="radio" aria-hidden />
              <div>
                <b>{MODE_NAMES[o.id] ?? o.id}</b>
                <span>{o.description}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>This conversation</h3>
        <dl className="kv">
          <dt>Answers</dt><dd>{answered.length}</dd>
          <dt>Total cost</dt><dd>{usd(cost)}</dd>
          <dt>If all on the large model</dt><dd>{usd(largeCost)}</dd>
          <dt>Saved</dt><dd style={{ color: "var(--accent)" }}>{largeCost > 0 ? pct(1 - cost / largeCost, 0) : "—"}</dd>
          <dt>Median first token</dt><dd>{ms(medianTtft)}</dd>
        </dl>
        <div style={{ marginTop: 14 }}>
          <div className="split-bar" role="img"
            aria-label={`${counts.small} small-model, ${counts.large} large-model, ${counts.cache} cached answers`}>
            <span style={{ width: `${(counts.small / total) * 100}%`, background: "var(--tier-small)" }} />
            <span style={{ width: `${(counts.large / total) * 100}%`, background: "var(--tier-large)" }} />
            <span style={{ width: `${(counts.cache / total) * 100}%`, background: "var(--tier-cache)" }} />
          </div>
          <div className="legend-row">
            <span><span className="swatch" style={{ background: "var(--tier-small)" }} /> Small {counts.small}</span>
            <span><span className="swatch" style={{ background: "var(--tier-large)" }} /> Large {counts.large}</span>
            <span><span className="swatch" style={{ background: "var(--tier-cache)" }} /> Cache {counts.cache}</span>
          </div>
        </div>
      </div>

      {config && (
        <div className="card">
          <h3>Models</h3>
          <dl className="kv" style={{ fontSize: 12.5 }}>
            <dt>Small</dt><dd title={config.tiers.small?.model}>{config.tiers.small?.model.split("/").pop()}</dd>
            <dt>Large</dt><dd title={config.tiers.large?.model}>{config.tiers.large?.model.split("/").pop()}</dd>
            <dt>Backend</dt><dd>{config.inference_mode}</dd>
          </dl>
          {config.inference_mode === "mock" && (
            <p className="card-sub" style={{ marginTop: 10 }}>
              Mock mode: answers and timings are simulated. Set <code>INFERENCE_MODE=nim</code> for real Nemotron models.
            </p>
          )}
        </div>
      )}
    </aside>
  );
}

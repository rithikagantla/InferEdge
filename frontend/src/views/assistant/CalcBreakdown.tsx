import type { AppConfig, RequestMetrics } from "../../types";
import { ms, num, pct, shortModel, titleCase, usd } from "../../lib/format";

/** "Show calculation": walks through how this reply's routing, token
 *  counts, cost and speed were derived, using the reply's own numbers. */
export function CalcBreakdown({ m, config }: { m: RequestMetrics; config: AppConfig | null }) {
  const tier = m.tier ?? "large";
  const tierInfo = config?.tiers[tier];
  const inPrice = m.input_price_per_1m ?? tierInfo?.input_cost_per_1m ?? 0;
  const outPrice = m.output_price_per_1m ?? tierInfo?.output_cost_per_1m ?? 0;
  const inCost = (m.input_tokens * inPrice) / 1e6;
  const outCost = (m.output_tokens * outPrice) / 1e6;
  const large = config?.tiers.large;
  const largeCost = large ? (m.input_tokens * large.input_cost_per_1m + m.output_tokens * large.output_cost_per_1m) / 1e6 : null;
  const savings = largeCost ? 1 - m.cost_usd / largeCost : null;
  const decodeMs = Math.max(m.latency_ms - m.ttft_ms, 0);
  const smallCats = new Set(config?.small_tier_categories ?? []);
  const estimated = config?.inference_mode === "mock";
  const scores = m.route_scores ? Object.entries(m.route_scores).sort((a, b) => b[1] - a[1]) : [];
  const maxScore = Math.max(...scores.map(([, s]) => s), 1e-9);
  let n = 0;

  return (
    <div className="calc" role="region" aria-label="How this answer was calculated">
      <div className="calc-head">
        <h4>How this answer was calculated</h4>
        <span className="badge">{m.mode} mode</span>
      </div>
      <div className="calc-steps">
        <Step n={++n} title="Which model answered, and why">
          {m.cached ? (
            <>This exact question was asked before in a single-turn chat, so the answer came from the
              LRU response cache. No model ran.</>
          ) : m.mode === "routing" && m.category ? (
            <>
              The semantic router embedded your message ({m.router_backend}) and compared it with each
              intent's centroid by cosine similarity. Highest score:{" "}
              <b>{titleCase(m.category)}</b> ({pct(m.route_confidence, 0)} confidence after softmax).
              {" "}{titleCase(m.category)} is {smallCats.has(m.category) ? "a routine intent, so it goes to the " : "a hard intent, so it goes to the "}
              <b>{tier} model</b>. Routing took {ms(m.router_time_ms, 3)}.
              {(m.route_confidence ?? 1) < 0.35 && config?.router_embeddings === "hashed" && (
                <> <span style={{ color: "var(--warn)" }}>Low confidence: the default hashed embeddings match words,
                  not meaning. <code>ROUTER_EMBEDDINGS=minilm</code> routes by meaning.</span></>
              )}
              <div className="mini-bars">
                {scores.map(([cat, s], i) => (
                  <div className={`mini-bar ${i === 0 ? "top" : ""}`} key={cat}>
                    <span className="muted">{titleCase(cat)}</span>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${Math.max((Math.max(s, 0) / maxScore) * 100, 2)}%` }} />
                    </div>
                    <span className="num" style={{ textAlign: "right" }}>{s.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <>In <b>{m.mode}</b> mode every message goes to the <b>{tier === "quantized" ? "large model (simulated FP8)" : "large model"}</b>{" "}
              (<span className="num">{shortModel(m.model)}</span>). No routing step.</>
          )}
        </Step>

        <Step n={++n} title="Tokens">
          Input = {estimated ? "" : "system prompt + "}{m.history_turns ? `${m.history_turns} earlier message${m.history_turns > 1 ? "s" : ""} + ` : ""}your message.
          Output = the reply. {estimated
            ? "The simulator has no tokenizer or system prompt, so both counts are estimated at about 4 characters per token."
            : "Counts are reported by the model server."}
          <div className="formula">
            <span className="k">input </span>{num(m.input_tokens)} tokens   <span className="k">output </span>{num(m.output_tokens)} tokens
          </div>
        </Step>

        <Step n={++n} title="Cost">
          {m.cached ? (
            <>A cache hit makes no model call, so its marginal cost is zero.
              <div className="formula">cost = <span className="r">$0</span></div></>
          ) : (
            <>
              Priced at the {tier} tier's rates per million tokens (assumed list prices; the free tier charges nothing).
              <div className="formula">
                <span className="k">input </span>{num(m.input_tokens)} × ${inPrice.toFixed(2)} / 1M = {usd(inCost)}<br />
                <span className="k">output</span> {num(m.output_tokens)} × ${outPrice.toFixed(2)} / 1M = {usd(outCost)}<br />
                <span className="k">total </span> <span className="r">{usd(m.cost_usd)}</span>
              </div>
            </>
          )}
        </Step>

        <Step n={++n} title="Speed">
          Time to first token is how long you waited before text appeared; the rest is generation.
          <div className="formula">
            <span className="k">first token </span>{ms(m.ttft_ms)}   <span className="k">generating </span>{ms(decodeMs)}   <span className="k">total </span>{ms(m.latency_ms)}<br />
            {m.cached ? (
              <><span className="k">tokens/sec </span>n/a (nothing was generated)</>
            ) : (
              <><span className="k">tokens/sec </span>{num(m.output_tokens)} ÷ {(m.latency_ms / 1000).toFixed(3)} s = <span className="r">{num(m.tokens_per_sec, 1)}</span></>
            )}
          </div>
        </Step>

        {largeCost != null && tier !== "large" && (
          <Step n={++n} title="Compared with always using the large model">
            The same tokens on the large tier (${large!.input_cost_per_1m.toFixed(2)} in / ${large!.output_cost_per_1m.toFixed(2)} out per 1M):
            <div className="formula">
              <span className="k">large model </span>{usd(largeCost)}   <span className="k">this answer </span>{usd(m.cost_usd)}<br />
              <span className="k">saved </span><span className="r">{pct(savings, 0)}</span>
            </div>
          </Step>
        )}
      </div>
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="calc-step">
      <span className="step-n" aria-hidden>{n}</span>
      <div>
        <div className="step-title">{title}</div>
        <div className="step-body">{children}</div>
      </div>
    </div>
  );
}

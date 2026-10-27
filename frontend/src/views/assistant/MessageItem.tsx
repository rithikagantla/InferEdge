import { useEffect, useState } from "react";
import type { AppConfig, ChatTurn } from "../../types";
import { Icon } from "../../components/Icon";
import { Markdown } from "../../components/Markdown";
import { ErrorNote, TierBadge } from "../../components/ui";
import { ms, titleCase, usd } from "../../lib/format";
import { usePrefersReducedMotion } from "../../hooks";
import { CalcBreakdown } from "./CalcBreakdown";

/** Reveals an answer word by word once it arrives. The answer is complete
 *  server-side; this only paces the display, capped at about 1.2 s. */
function Reveal({ text, onDone }: { text: string; onDone: () => void }) {
  const words = text.split(/(\s+)/);
  const [count, setCount] = useState(0);
  useEffect(() => {
    const step = Math.max(1, Math.ceil(words.length / 80));
    const id = window.setInterval(() => {
      setCount((c) => {
        const next = c + step;
        if (next >= words.length) {
          window.clearInterval(id);
          window.setTimeout(onDone, 0);
        }
        return next;
      });
    }, 15);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);
  return (
    <div className="answer">
      <p style={{ whiteSpace: "pre-wrap" }}>
        {words.slice(0, count).join("")}
        <span className="caret" aria-hidden />
      </p>
    </div>
  );
}

export function MessageItem({ turn, config, onRetry, onRevealed }: {
  turn: ChatTurn;
  config: AppConfig | null;
  onRetry: () => void;
  onRevealed: () => void;
}) {
  const [showCalc, setShowCalc] = useState(false);
  const [copied, setCopied] = useState(false);
  const reduced = usePrefersReducedMotion();

  if (turn.role === "user") {
    return (
      <div className="msg user">
        <div className="msg-body"><div className="bubble-user">{turn.content}</div></div>
      </div>
    );
  }

  const m = turn.metrics;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(turn.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard blocked */
    }
  };

  return (
    <div className="msg">
      <div className="avatar" aria-hidden>IE</div>
      <div className="msg-body">
        {turn.pending ? (
          <div aria-live="polite">
            <span className="typing" aria-hidden><span /><span /><span /></span>
            <span className="typing-label">
              {turn.mode === "routing" ? "Routing and generating…" : "Generating…"}
            </span>
          </div>
        ) : turn.error ? (
          <ErrorNote message={turn.error} onRetry={onRetry} />
        ) : turn.animate && !reduced ? (
          <Reveal text={turn.content} onDone={onRevealed} />
        ) : (
          <Markdown text={turn.content} />
        )}

        {m && !turn.pending && !turn.error && (
          <>
            <div className="msg-meta">
              <TierBadge tier={m.tier} cached={m.cached} />
              {m.category && <span className="badge">{titleCase(m.category)}</span>}
              <span className="meta-item" title="Time to first token">TTFT {ms(m.ttft_ms)}</span>
              <span className="meta-sep">·</span>
              <span className="meta-item" title="Total latency">{ms(m.latency_ms)}</span>
              <span className="meta-sep">·</span>
              <span className="meta-item" title="Estimated cost">{usd(m.cost_usd)}</span>
              <div className="msg-actions">
                <button
                  type="button"
                  className="btn btn-sm btn-ghost"
                  aria-expanded={showCalc}
                  onClick={() => setShowCalc((v) => !v)}
                >
                  <Icon name="calc" size={14} /> {showCalc ? "Hide calculation" : "Show calculation"}
                </button>
                <button type="button" className="btn btn-sm btn-ghost icon-btn" onClick={copy} aria-label="Copy answer">
                  <Icon name={copied ? "check" : "copy"} size={14} />
                </button>
              </div>
            </div>
            {showCalc && <CalcBreakdown m={m} config={config} />}
          </>
        )}
      </div>
    </div>
  );
}

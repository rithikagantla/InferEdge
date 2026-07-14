import { useEffect, useState } from "react";
import { api } from "../api";
import type { ChatResponse, OptimizationMode } from "../types";
import { OPTIMIZATION_MODES } from "../types";

export default function ChatPanel({ onActivity }: { onActivity: () => void }) {
  const [prompt, setPrompt] = useState("My order is delayed. Can you help?");
  const [mode, setMode] = useState<OptimizationMode>("baseline");
  const [samples, setSamples] = useState<string[]>([]);
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.samplePrompts().then((d) => setSamples(d.prompts.slice(0, 6))).catch(() => {});
  }, []);

  const send = async () => {
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.chat(prompt.trim(), mode));
      onActivity();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const m = result?.metrics;
  return (
    <div className="card">
      <h2>Live chat</h2>
      <p className="hint">
        Send a support prompt through a chosen optimization mode and inspect
        per-request latency, tokens, and cost.
      </p>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) send();
        }}
        placeholder="Ask a customer-support question…"
        aria-label="Support prompt"
      />
      <div className="row">
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as OptimizationMode)}
          aria-label="Optimization mode"
        >
          {OPTIMIZATION_MODES.filter((x) => x !== "batching").map((x) => (
            <option key={x} value={x}>
              {x}
            </option>
          ))}
        </select>
        <button onClick={send} disabled={busy || !prompt.trim()}>
          {busy ? "Generating…" : "Send"}
        </button>
      </div>
      <div className="chips">
        {samples.map((s) => (
          <button key={s} className="chip" onClick={() => setPrompt(s)}>
            {s}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}
      {result && m && (
        <>
          <div className="answer">{result.response}</div>
          <div className="kv">
            <div>latency <b>{m.latency_ms.toFixed(1)}ms</b></div>
            <div>ttft <b>{m.ttft_ms.toFixed(1)}ms</b></div>
            <div>tok/s <b>{m.tokens_per_sec.toFixed(1)}</b></div>
            <div>tokens <b>{m.input_tokens}→{m.output_tokens}</b></div>
            <div>cost <b>${m.cost_usd.toFixed(6)}</b></div>
            {m.cached && <div><b>cache hit</b></div>}
            {m.category && <div>intent <b>{m.category}</b></div>}
          </div>
          <p className="note">model: {m.model}</p>
        </>
      )}
    </div>
  );
}

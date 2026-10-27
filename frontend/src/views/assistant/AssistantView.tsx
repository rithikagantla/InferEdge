import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../api";
import { Icon } from "../../components/Icon";
import { useAsync, useLocalStorage } from "../../hooks";
import { SAMPLE_CATEGORIES } from "../../lib/series";
import { titleCase } from "../../lib/format";
import type { AppConfig, ChatMode, ChatTurn } from "../../types";
import { MessageItem } from "./MessageItem";
import { SessionPanel } from "./SessionPanel";

const MAX_HISTORY = 20;
const newId = () => Math.random().toString(36).slice(2, 10);

export function AssistantView({ config, onActivity }: { config: AppConfig | null; onActivity: () => void }) {
  const [turns, setTurns] = useLocalStorage<ChatTurn[]>("inferedge.chat.v1", []);
  const [mode, setMode] = useLocalStorage<ChatMode>("inferedge.chat.mode", "routing");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const samples = useAsync(() => api.samplePrompts());

  // Restored from storage: a request in flight when the page closed is gone,
  // and finished answers shouldn't replay their reveal.
  useEffect(() => {
    setTurns((ts) => ts.map((t) => (t.pending
      ? { ...t, pending: false, error: "This answer was interrupted. Retry to ask again." }
      : t.animate ? { ...t, animate: false } : t)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // A mode saved from an earlier session may not be offered by this backend.
  useEffect(() => {
    if (config && !config.chat_modes.some((m) => m.id === mode)) setMode("routing");
  }, [config, mode, setMode]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns.length, turns[turns.length - 1]?.pending]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [draft]);

  const ask = useCallback(async (prompt: string, prior: ChatTurn[], replaceId?: string) => {
    const history = prior
      .filter((t) => !t.pending && !t.error && t.content)
      .slice(-MAX_HISTORY)
      .map((t) => ({ role: t.role, content: t.content }));
    const pendingId = replaceId ?? newId();
    const pending: ChatTurn = { id: pendingId, role: "assistant", content: "", pending: true, mode, createdAt: Date.now() };
    setTurns((ts) => (replaceId ? ts.map((t) => (t.id === replaceId ? pending : t)) : [...ts, pending]));
    setBusy(true);
    try {
      const res = await api.chat(prompt, mode, history);
      setTurns((ts) => ts.map((t) => (t.id === pendingId
        ? { ...t, pending: false, content: res.response, metrics: res.metrics, animate: true }
        : t)));
      onActivity();
    } catch (e) {
      setTurns((ts) => ts.map((t) => (t.id === pendingId
        ? { ...t, pending: false, error: e instanceof Error ? e.message : String(e) }
        : t)));
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }, [mode, onActivity, setTurns]);

  const send = (text?: string) => {
    const prompt = (text ?? draft).trim();
    if (!prompt || busy) return;
    const user: ChatTurn = { id: newId(), role: "user", content: prompt, createdAt: Date.now() };
    const prior = turns;
    setTurns((ts) => [...ts, user]);
    setDraft("");
    void ask(prompt, prior);
  };

  const retry = (assistantId: string) => {
    const idx = turns.findIndex((t) => t.id === assistantId);
    const userTurn = [...turns.slice(0, idx)].reverse().find((t) => t.role === "user");
    if (!userTurn || busy) return;
    void ask(userTurn.content, turns.slice(0, turns.indexOf(userTurn)), assistantId);
  };

  const markRevealed = (id: string) =>
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, animate: false } : t)));

  const suggestions = (samples.data?.prompts ?? [])
    .filter((_, i) => i % 5 === 0)
    .slice(0, 6)
    .map((p, i) => ({ prompt: p, category: SAMPLE_CATEGORIES[i] }));

  return (
    <>
      <header className="page-header">
        <div>
          <h1 className="page-title">Support assistant</h1>
          <p className="page-sub">
            Chat with the customer-support agent. Every answer shows which model handled it, what it
            cost and how fast it was; open <b>Show calculation</b> to see how each number was derived.
          </p>
        </div>
        <button className="btn" type="button" onClick={() => setTurns([])} disabled={!turns.length || busy}>
          <Icon name="plus" size={15} /> New chat
        </button>
      </header>

      <div className="assistant">
        <section className="card chat-card" aria-label="Conversation">
          <div className="chat-scroll" ref={scrollRef} aria-live="polite">
            {turns.length === 0 ? (
              <div className="chat-empty">
                <div className="avatar" style={{ width: 44, height: 44, margin: "0 auto", fontSize: 16 }}>IE</div>
                <h2>How can I help today?</h2>
                <p>Try a question below. Routine requests go to the fast small model; technical issues and escalations go to the large one.</p>
                <div className="suggestions">
                  {suggestions.map((s) => (
                    <button key={s.prompt} type="button" className="suggestion" onClick={() => send(s.prompt)}>
                      <span className="cat">{titleCase(s.category)}</span>
                      <span className="txt">{s.prompt}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              turns.map((t) => (
                <MessageItem
                  key={t.id}
                  turn={t}
                  config={config}
                  onRetry={() => retry(t.id)}
                  onRevealed={() => markRevealed(t.id)}
                />
              ))
            )}
          </div>

          <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
            <div className="composer-row">
              <label className="sr-only" htmlFor="composer">Message</label>
              <textarea
                id="composer"
                ref={inputRef}
                rows={1}
                value={draft}
                maxLength={4000}
                placeholder="Describe the customer's issue…"
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    send();
                  }
                }}
              />
              <button className="btn btn-primary" type="submit" disabled={busy || !draft.trim()} aria-label="Send message">
                {busy ? <span className="spinner" aria-hidden /> : <Icon name="send" size={15} />}
                <span>Send</span>
              </button>
            </div>
            <div className="composer-foot">
              <span><span className="kbd">Enter</span> to send · <span className="kbd">Shift</span>+<span className="kbd">Enter</span> for a new line</span>
              <span className="num">{draft.length}/4000</span>
            </div>
          </form>
        </section>

        <SessionPanel turns={turns} config={config} mode={mode} onMode={setMode} />
      </div>
    </>
  );
}

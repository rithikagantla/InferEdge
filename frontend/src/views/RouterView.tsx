import { useEffect, useState } from "react";
import { api } from "../api";
import { Icon } from "../components/Icon";
import { Card, ErrorNote, Stat } from "../components/ui";
import { useDebounced } from "../hooks";
import { ms, num, pct, titleCase } from "../lib/format";
import type { AppConfig, RouteResponse, RouterBenchmark } from "../types";

const LOW_CONFIDENCE = 0.35;

const EXAMPLES = [
  "The courier says delivered but nothing is at my door.",
  "Please delete my account and all my data.",
  "The SDK throws a null pointer exception on startup.",
  "I'm not talking to a chatbot about this, put a person on.",
  "Is the restocking fee taken out of my reimbursement?",
];

export function RouterView({ config }: { config: AppConfig | null }) {
  const [text, setText] = useState(EXAMPLES[0]);
  const debounced = useDebounced(text.trim(), 250);
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bench, setBench] = useState<RouterBenchmark | null>(null);
  const [benching, setBenching] = useState(false);

  useEffect(() => {
    if (!debounced) return;
    let cancelled = false;
    api.route(debounced)
      .then((r) => { if (!cancelled) { setRoute(r); setError(null); } })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, [debounced]);

  const runBench = async () => {
    setBenching(true);
    try {
      setBench(await api.routerBenchmark(20000));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBenching(false);
    }
  };

  const scores = route ? Object.entries(route.scores).sort((a, b) => b[1] - a[1]) : [];
  const max = Math.max(...scores.map(([, s]) => s), 1e-9);
  const small = route?.model_tier === "small";

  return (
    <>
      <header className="page-header">
        <div>
          <h1 className="page-title">Semantic router</h1>
          <p className="page-sub">
            Type a customer message and watch the router classify it live. Routine intents go to the
            small model; technical issues and escalations go to the large one.
          </p>
        </div>
      </header>
      <div className="page-body">
        <div className="grid-2">
          <Card title="Customer message" sub="Classified as you type">
            <label className="sr-only" htmlFor="router-input">Customer message</label>
            <textarea
              id="router-input"
              className="router-input"
              value={text}
              maxLength={4000}
              onChange={(e) => setText(e.target.value)}
            />
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
              {EXAMPLES.map((ex) => (
                <button key={ex} type="button" className="chip" onClick={() => setText(ex)}>{ex}</button>
              ))}
            </div>
            {error && <div style={{ marginTop: 12 }}><ErrorNote message={error} /></div>}
          </Card>

          <Card title="Routing decision" sub={route ? `${route.backend} · ${ms(route.route_time_ms, 3)}` : "Waiting for input"}>
            {route ? (
              <>
                <div className="route-hero">
                  <span className="route-cat">{titleCase(route.category)}</span>
                  <span className="arrow"><Icon name="chevron" size={18} /></span>
                  <span className={`badge ${small ? "small" : "large"}`} style={{ fontSize: 13, padding: "4px 12px" }}>
                    {small ? "Small model" : "Large model"}
                  </span>
                  {route.confidence < LOW_CONFIDENCE && <span className="badge cache">Low confidence</span>}
                </div>
                <p className="card-sub" style={{ margin: "8px 0 16px" }}>
                  {pct(route.confidence, 0)} confidence. {small
                    ? "A routine intent: the fast, cheap model handles it."
                    : "A hard or sensitive intent: the strong model handles it."}{" "}
                  Model: <span className="num">{config?.tiers[route.model_tier]?.model.split("/").pop() ?? route.model_tier}</span>
                </p>
                <div className="bars">
                  {scores.map(([cat, s], i) => (
                    <div className="bar-row" key={cat}>
                      <span className="label">{titleCase(cat)}</span>
                      <div className="bar-track">
                        <div className={`bar-fill ${i === 0 ? "win" : ""}`} style={{ width: `${Math.max((Math.max(s, 0) / max) * 100, 1.5)}%` }} />
                      </div>
                      <span className="val">{s.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
                <p className="card-sub" style={{ marginTop: 12 }}>
                  Cosine similarity between the message's embedding and each intent's centroid.
                </p>
                {route.confidence < LOW_CONFIDENCE && config?.router_embeddings === "hashed" && (
                  <p className="card-sub" style={{ marginTop: 8, color: "var(--warn)" }}>
                    The scores are close, so this pick is a guess. The default hashed embeddings match
                    words, not meaning (42% category accuracy on held-out messages). Start the backend
                    with <code>ROUTER_EMBEDDINGS=minilm</code> for real sentence embeddings (93%).
                  </p>
                )}
              </>
            ) : (
              <p className="muted">Type a message to see how it would be routed.</p>
            )}
          </Card>
        </div>

        <Card
          title="CPU vs GPU scoring"
          sub="Scores 20,000 random intent embeddings against the six centroids on this server"
          actions={
            <button className="btn" type="button" onClick={runBench} disabled={benching}>
              {benching ? <span className="spinner" aria-hidden /> : <Icon name="gpu" size={15} />} Run benchmark
            </button>
          }
        >
          {bench ? (
            <div className="stats">
              <Stat label="CPU (NumPy)" value={ms(bench.cpu_time_ms, 2)} />
              <Stat label="GPU (CuPy)" value={bench.gpu_time_ms != null ? ms(bench.gpu_time_ms, 2) : "No GPU"} />
              <Stat label="Speedup" value={bench.speedup != null ? `${num(bench.speedup, 1)}×` : "—"} foot={bench.backend} />
            </div>
          ) : (
            <p className="muted" style={{ margin: 0 }}>
              This machine has {config?.router_backend?.includes("cuda") ? "an NVIDIA GPU" : "no NVIDIA GPU, so it runs in CPU fallback mode"}.
              Measured results from a cloud NVIDIA L4, including real MiniLM embeddings, are on the Results page.
            </p>
          )}
        </Card>
      </div>
    </>
  );
}

import type { Theme } from "../hooks";
import type { AppConfig, MetricsSnapshot } from "../types";
import { Icon } from "./Icon";

export type View = "assistant" | "router" | "benchmarks" | "results";

const NAV: { id: View; label: string; icon: string; hint: string }[] = [
  { id: "assistant", label: "Assistant", icon: "chat", hint: "Chat" },
  { id: "router", label: "Router", icon: "route", hint: "Live" },
  { id: "benchmarks", label: "Benchmarks", icon: "bench", hint: "Run" },
  { id: "results", label: "Results", icon: "results", hint: "Measured" },
];

export function Sidebar({ view, onView, config, snapshot, online, theme, onToggleTheme }: {
  view: View;
  onView: (v: View) => void;
  config: AppConfig | null;
  snapshot: MetricsSnapshot | null;
  online: boolean;
  theme: Theme;
  onToggleTheme: () => void;
}) {
  return (
    <nav className="sidebar" aria-label="Main">
      <div className="brand">
        <div className="brand-mark" aria-hidden>IE</div>
        <div>
          <div className="brand-name">InferEdge</div>
          <div className="brand-sub">LLM inference platform</div>
        </div>
      </div>

      <div className="nav">
        {NAV.map((n) => (
          <button
            key={n.id}
            type="button"
            className="nav-item"
            aria-current={view === n.id ? "page" : undefined}
            onClick={() => onView(n.id)}
          >
            <Icon name={n.icon} size={17} />
            {n.label}
            <span className="nav-hint">{n.hint}</span>
          </button>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="status-card" aria-live="polite">
          <div className="status-row">
            <span>Backend</span>
            <span><span className={`dot ${online ? "ok" : "bad"}`} />{online ? "Online" : "Offline"}</span>
          </div>
          <div className="status-row"><span>Inference</span><span>{config?.inference_mode ?? "—"}</span></div>
          <div className="status-row">
            <span>Router</span>
            <span title={config?.router_backend}>{config?.router_backend?.split(" (")[0] ?? "—"}</span>
          </div>
          <div className="status-row"><span>Requests</span><span className="num">{snapshot?.total_requests ?? 0}</span></div>
        </div>
        <button
          type="button"
          className="btn btn-ghost"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          <Icon name={theme === "dark" ? "sun" : "moon"} size={15} />
          {theme === "dark" ? "Light" : "Dark"} theme
        </button>
      </div>
    </nav>
  );
}

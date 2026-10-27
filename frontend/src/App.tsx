import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { Sidebar, type View } from "./components/Sidebar";
import { useAsync, useTheme } from "./hooks";
import { useThemeColors } from "./lib/useThemeColors";
import type { MetricsSnapshot } from "./types";
import { AssistantView } from "./views/assistant/AssistantView";
import { RouterView } from "./views/RouterView";

// Charts (Recharts) are only needed on these two views, so they load on demand
// and the assistant stays fast to open.
const BenchmarksView = lazy(() => import("./views/BenchmarksView").then((m) => ({ default: m.BenchmarksView })));
const ResultsView = lazy(() => import("./views/results/ResultsView").then((m) => ({ default: m.ResultsView })));

const VIEWS: View[] = ["assistant", "router", "benchmarks", "results"];
const fromHash = (): View => {
  const v = window.location.hash.replace("#/", "") as View;
  return VIEWS.includes(v) ? v : "assistant";
};

export default function App() {
  const { theme, toggle } = useTheme();
  const colors = useThemeColors(theme);
  const [view, setView] = useState<View>(fromHash);
  const [snapshot, setSnapshot] = useState<MetricsSnapshot | null>(null);
  const [online, setOnline] = useState(true);
  const config = useAsync(() => api.config());

  useEffect(() => {
    const onHash = () => setView(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (v: View) => {
    window.location.hash = `/${v}`;
    setView(v);
  };

  const refresh = useCallback(() => {
    api.metrics()
      .then((s) => { setSnapshot(s); setOnline(true); })
      .catch(() => setOnline(false));
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 5000);
    return () => window.clearInterval(id);
  }, [refresh]);

  // Coming back online after a failed first load: fetch the config again.
  useEffect(() => {
    if (online && config.error) void config.reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [online]);

  useEffect(() => {
    const titles: Record<View, string> = {
      assistant: "Assistant", router: "Router", benchmarks: "Benchmarks", results: "Results",
    };
    document.title = `${titles[view]} · InferEdge`;
  }, [view]);

  return (
    <div className="shell">
      <Sidebar view={view} onView={go} config={config.data} snapshot={snapshot} online={online}
        theme={theme} onToggleTheme={toggle} />
      <main className="main" id="main">
        {view === "assistant" && <AssistantView config={config.data} onActivity={refresh} />}
        {view === "router" && <RouterView config={config.data} />}
        <Suspense fallback={<div className="page-body" style={{ paddingTop: 28 }}><p className="muted">Loading…</p></div>}>
          {view === "benchmarks" && <BenchmarksView snapshot={snapshot} colors={colors} onActivity={refresh} />}
          {view === "results" && <ResultsView colors={colors} />}
        </Suspense>
      </main>
    </div>
  );
}

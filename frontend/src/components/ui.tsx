import type { ReactNode } from "react";
import { Icon } from "./Icon";

export function Card({ title, sub, actions, children, className = "" }: {
  title?: ReactNode;
  sub?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card card-pad ${className}`}>
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {sub && <p className="card-sub">{sub}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T;
  options: { id: T; label: ReactNode }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.id} type="button" aria-pressed={o.id === value} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Stat({ label, value, unit, foot, icon }: {
  label: string;
  value: ReactNode;
  unit?: string;
  foot?: ReactNode;
  icon?: string;
}) {
  return (
    <div className="card stat">
      <div className="stat-label">{icon && <Icon name={icon} size={13} />}{label}</div>
      <div className="stat-value num">
        {value}
        {unit && <small>{unit}</small>}
      </div>
      {foot && <div className="stat-foot">{foot}</div>}
    </div>
  );
}

export function TierBadge({ tier, cached }: { tier: string | null; cached?: boolean }) {
  if (cached) return <span className="badge cache"><Icon name="bolt" size={11} />Cache hit</span>;
  if (tier === "small") return <span className="badge small">Small model</span>;
  if (tier === "quantized") return <span className="badge large">Large · FP8 (sim)</span>;
  return <span className="badge large">Large model</span>;
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="alert" role="alert">
      <Icon name="info" size={16} />
      <div style={{ flex: 1 }}>{message}</div>
      {onRetry && (
        <button className="btn btn-sm btn-ghost" onClick={onRetry} type="button">
          <Icon name="retry" size={13} /> Retry
        </button>
      )}
    </div>
  );
}

export function Bars({ rows, format, highlightMax = true }: {
  rows: { label: string; value: number }[];
  format: (v: number) => string;
  highlightMax?: boolean;
}) {
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  const top = rows.reduce((best, r) => (r.value > best.value ? r : best), rows[0]);
  return (
    <div className="bars">
      {rows.map((r) => (
        <div className="bar-row" key={r.label}>
          <span className="label" title={r.label}>{r.label}</span>
          <div className="bar-track">
            <div
              className={`bar-fill ${highlightMax && r === top ? "win" : ""}`}
              style={{ width: `${Math.max((r.value / max) * 100, 1.5)}%` }}
            />
          </div>
          <span className="val">{format(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

interface Bar {
  name: string;
  value: number;
}

interface Props {
  title: string;
  bars: Bar[];
  format?: (v: number) => string;
}

/** Single-hue horizontal bar comparison (identity lives on the axis
 *  labels, so no legend/palette is needed). */
export default function BarChart({ title, bars, format }: Props) {
  const max = Math.max(...bars.map((b) => b.value), 1e-9);
  const fmt = format ?? ((v: number) => v.toFixed(2));
  return (
    <div>
      <p className="chart-title">{title}</p>
      <div className="bars" role="img" aria-label={title}>
        {bars.map((b) => (
          <div className="bar-row" key={b.name}>
            <span className="name">{b.name}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${Math.max((b.value / max) * 100, 1)}%` }}
              />
            </div>
            <span className="num">{fmt(b.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

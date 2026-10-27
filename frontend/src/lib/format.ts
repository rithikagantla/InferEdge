export function ms(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (v === 0) return "0 ms";
  if (v >= 10_000) return `${(v / 1000).toFixed(1)} s`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  return `${v.toFixed(v < 10 && digits === 0 ? 1 : digits)} ms`;
}

/** Dollar amounts that are often fractions of a cent. */
export function usd(v: number | null | undefined): string {
  if (v == null) return "—";
  if (v === 0) return "$0";
  if (v < 0.0001) return `$${v.toFixed(7)}`;
  // Keep sub-dollar amounts at one precision so they compare at a glance.
  if (v < 1) return `$${v.toFixed(5)}`;
  return `$${v.toFixed(2)}`;
}

export function num(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function pct(v: number | null | undefined, digits = 0): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** "nvidia/nemotron-3-ultra-550b-a55b" -> "nemotron-3-ultra-550b-a55b" */
export function shortModel(model: string): string {
  return model.replace(/^mock\//, "").split("/").pop() ?? model;
}

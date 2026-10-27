import { useEffect, useState } from "react";
import type { Theme } from "../hooks";

export interface ThemeColors {
  series: string[];
  ink: string;
  ink2: string;
  ink3: string;
  grid: string;
  border: string;
  accent: string;
  accentFill: string;
  surface: string;
}

const read = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Charts set colors as SVG attributes, so resolve the CSS tokens to real
 *  values (re-read whenever the theme changes). */
export function useThemeColors(theme: Theme): ThemeColors {
  const resolve = (): ThemeColors => ({
    series: [1, 2, 3, 4, 5, 6].map((i) => read(`--series-${i}`)),
    ink: read("--ink"),
    ink2: read("--ink-2"),
    ink3: read("--ink-3"),
    grid: read("--grid"),
    border: read("--border-strong"),
    accent: read("--accent"),
    accentFill: read("--accent-fill"),
    surface: read("--surface"),
  });
  const [colors, setColors] = useState<ThemeColors>(resolve);
  useEffect(() => {
    // The data-theme attribute is set in an effect too; read after it lands.
    const id = window.requestAnimationFrame(() => setColors(resolve()));
    return () => window.cancelAnimationFrame(id);
  }, [theme]);
  return colors;
}

const PATHS: Record<string, string> = {
  chat: "M4 5h16v11H8l-4 4V5z",
  route: "M6 3v6a3 3 0 0 0 3 3h6a3 3 0 0 1 3 3v6M6 21v-4M18 3v4M3 6l3-3 3 3M15 18l3 3 3-3",
  bench: "M4 20V10M10 20V4M16 20v-7M22 20H2",
  results: "M3 3v18h18M7 15l4-4 3 3 6-7",
  send: "M4 12l16-8-6 16-3-7-7-1z",
  plus: "M12 5v14M5 12h14",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  check: "M5 13l4 4L19 7",
  calc: "M6 3h12v18H6zM9 7h6M9 11h1M12 11h1M15 11h1M9 14h1M12 14h1M15 14h1M9 17h1M12 17h1M15 17h1",
  chevron: "M9 6l6 6-6 6",
  sun: "M12 4V2M12 22v-2M4 12H2M22 12h-2M5.6 5.6L4.2 4.2M19.8 19.8l-1.4-1.4M5.6 18.4l-1.4 1.4M19.8 4.2l-1.4 1.4M12 7a5 5 0 1 0 0 10a5 5 0 0 0 0-10z",
  moon: "M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z",
  retry: "M4 4v6h6M20 20v-6h-6M5 15a7 7 0 0 0 12 2M19 9A7 7 0 0 0 7 7",
  bolt: "M13 2L4 14h7l-1 8 9-12h-7l1-8z",
  info: "M12 8h.01M11 12h1v5h1M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18z",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  gpu: "M3 7h18v10H3zM7 17v3M12 17v3M17 17v3M7 11h4v2H7zM14 11h3v2h-3z",
};

export function Icon({ name, size = 16, label }: { name: keyof typeof PATHS | string; size?: number; label?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <path d={PATHS[name] ?? PATHS.info} />
    </svg>
  );
}

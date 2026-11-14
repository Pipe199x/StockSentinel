"use client";

/**
 * Custom: the Aceternity registry has no data-chart primitive, so price and equity
 * charts are plain responsive SVG styled to match the Aceternity cards.
 */
import { useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { fin, num, shortDate } from "@/lib/format";

const monthYear = (iso?: string | null) =>
  iso ? new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(iso.slice(0, 10) + "T00:00:00Z")) : "—";

export type Point = { date: string; value: number };
export type Level = { value: number; label: string; tone: "gain" | "loss" | "muted" };

const W = 720;
const H = 260;
const PAD = { l: 8, r: 64, t: 14, b: 26 };

function ticks(lo: number, hi: number, n = 4) {
  const step = (hi - lo) / n;
  return Array.from({ length: n + 1 }, (_, i) => lo + i * step);
}

export function LineChart({
  points,
  levels = [],
  format = (v: number) => num(v),
  area = false,
  className,
  label,
}: {
  points: Point[];
  levels?: Level[];
  format?: (v: number) => string;
  area?: boolean;
  className?: string;
  label: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const geo = useMemo(() => {
    const vals = points.map((p) => p.value).filter(fin);
    const lv = levels.map((l) => l.value).filter(fin);
    if (!vals.length) return null;
    let lo = Math.min(...vals, ...lv);
    let hi = Math.max(...vals, ...lv);
    if (hi === lo) {
      hi += 1;
      lo -= 1;
    }
    const padY = (hi - lo) * 0.06;
    lo -= padY;
    hi += padY;
    const x = (i: number) => PAD.l + (i / Math.max(points.length - 1, 1)) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);
    const d = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join("");
    const a = `${d}L${x(points.length - 1).toFixed(1)},${H - PAD.b}L${x(0).toFixed(1)},${H - PAD.b}Z`;
    return { lo, hi, x, y, d, a };
  }, [points, levels]);

  if (!geo) {
    return <div className="grid h-48 place-items-center text-sm text-neutral-500">No data yet</div>;
  }
  const last = points[points.length - 1];
  const up = points.length > 1 && last.value >= points[0].value;
  const stroke = up ? "stroke-emerald-500" : "stroke-rose-500";
  const hp = hover != null ? points[hover] : null;

  const onMove = (e: React.PointerEvent) => {
    const r = svgRef.current?.getBoundingClientRect();
    if (!r) return;
    const xr = ((e.clientX - r.left) / r.width) * W;
    const i = Math.round(((xr - PAD.l) / (W - PAD.l - PAD.r)) * (points.length - 1));
    setHover(Math.min(points.length - 1, Math.max(0, i)));
  };

  // Set: with 1-2 points the first/mid/last indices collide (duplicate keys).
  const dateTicks = [...new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])];
  // Charts longer than ~10 months label month + year ("Feb 2019"); shorter ones day + month.
  const spanDays = (Date.parse(last.date) - Date.parse(points[0].date)) / 864e5;
  const tickDate = spanDays > 300 ? monthYear : shortDate;

  return (
    <div className={cn("relative", className)}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full touch-none select-none"
        role="img"
        aria-label={label}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id="lc-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={up ? "#10b981" : "#f43f5e"} stopOpacity="0.25" />
            <stop offset="100%" stopColor={up ? "#10b981" : "#f43f5e"} stopOpacity="0" />
          </linearGradient>
        </defs>
        {ticks(geo.lo, geo.hi).map((t, i) => (
          <g key={`${i}-${t}`}>
            <line x1={PAD.l} x2={W - PAD.r} y1={geo.y(t)} y2={geo.y(t)} className="stroke-neutral-200 dark:stroke-white/[0.06]" />
            <text x={W - PAD.r + 6} y={geo.y(t)} dominantBaseline="middle" className="fill-neutral-500 text-[11px]">
              {format(t)}
            </text>
          </g>
        ))}
        {dateTicks.map((i) => (
          <text key={i} x={geo.x(i)} y={H - 6} textAnchor={i === 0 ? "start" : i === points.length - 1 ? "end" : "middle"} className="fill-neutral-500 text-[11px]">
            {tickDate(points[i]?.date)}
          </text>
        ))}
        {area && <path d={geo.a} fill="url(#lc-fill)" />}
        {levels.filter((l) => fin(l.value)).map((l) => (
          <g key={l.label}>
            <line
              x1={PAD.l}
              x2={W - PAD.r}
              y1={geo.y(l.value)}
              y2={geo.y(l.value)}
              strokeDasharray="5 5"
              className={cn(l.tone === "gain" ? "stroke-emerald-500" : l.tone === "loss" ? "stroke-rose-500" : "stroke-neutral-400")}
            />
            <text x={W - PAD.r - 4} y={geo.y(l.value) - 5} textAnchor="end" className={cn("text-[11px] font-medium", l.tone === "gain" ? "fill-emerald-500" : l.tone === "loss" ? "fill-rose-500" : "fill-neutral-400")}>
              {l.label} {format(l.value)}
            </text>
          </g>
        ))}
        <path d={geo.d} fill="none" strokeWidth={2} strokeLinejoin="round" className={stroke} />
        {hp && (
          <g>
            <line x1={geo.x(hover!)} x2={geo.x(hover!)} y1={PAD.t} y2={H - PAD.b} className="stroke-neutral-400/60" />
            <circle cx={geo.x(hover!)} cy={geo.y(hp.value)} r={4} className={cn("fill-white dark:fill-[#0e0e0c]", stroke)} strokeWidth={2} />
          </g>
        )}
      </svg>
      {hp && (
        <div className="pointer-events-none absolute top-2 left-2 rounded-md border border-neutral-200 bg-white/90 px-2 py-1 text-xs shadow-sm backdrop-blur dark:border-white/10 dark:bg-neutral-900/90">
          <span className="font-semibold">{format(hp.value)}</span>
          <span className="ml-2 text-neutral-500">{shortDate(hp.date)}</span>
        </div>
      )}
    </div>
  );
}

/** Custom: horizontal bars for a list of values (CPCV paths, folds). */
export function Bars({ items, format, threshold }: { items: { label: string; value: number }[]; format: (v: number) => string; threshold?: number }) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), threshold ?? 0, 1e-9);
  return (
    <ul className="space-y-2">
      {items.map((it) => (
        <li key={it.label} className="grid grid-cols-[5rem_1fr_3.75rem] items-center gap-2 text-xs sm:grid-cols-[6.5rem_1fr_4.5rem] sm:gap-3">
          <span className="truncate text-neutral-500">{it.label}</span>
          <span className="relative h-2 rounded-full bg-neutral-200 dark:bg-white/[0.06]">
            <span
              className={cn("absolute inset-y-0 left-0 rounded-full", it.value >= 0 ? "bg-emerald-500/80" : "bg-rose-500/80")}
              style={{ width: `${(Math.abs(it.value) / max) * 100}%` }}
            />
          </span>
          <span className="text-right tabular-nums">{format(it.value)}</span>
        </li>
      ))}
    </ul>
  );
}

import { useId, type CSSProperties } from "react";
import { cn } from "@/lib/utils";
import { OceanMark, OceanMarkDefs } from "./ocean-mark";
import { MARK_BOX } from "./ocean-mark-geometry";

/**
 * Ocean Data Group watermark.
 *
 * Custom component: the Aceternity registry has no watermark primitive.
 *
 * Fills its nearest positioned ancestor (give the parent `relative`), or the
 * viewport with `fixed`. Never intercepts clicks or text selection, and is
 * hidden from assistive tech.
 *
 *   <section className="relative">
 *     ...content...
 *     <Watermark variant="tile" tone="white" />
 *   </section>
 */

type Variant = "corner" | "center" | "tile";
/** white = for dark backgrounds, navy = for light backgrounds, auto = follows color scheme. */
type Tone = "white" | "navy" | "auto";
type Corner = "top-left" | "top-right" | "bottom-left" | "bottom-right";

export type WatermarkProps = {
  variant?: Variant;
  tone?: Tone;
  /** 0-1. Defaults: corner 0.55, center 0.07, tile 0.06. */
  opacity?: number;
  /** Corner variant only. */
  position?: Corner;
  /** Show the "Ocean Data Group" wordmark (corner and center variants). */
  label?: boolean;
  /** Pin to the viewport instead of the positioned parent. */
  fixed?: boolean;
  className?: string;
  style?: CSSProperties;
};

const DEFAULT_OPACITY: Record<Variant, number> = {
  corner: 0.55,
  center: 0.07,
  tile: 0.06,
};

const TONE_CLASS: Record<Tone, string> = {
  white: "text-white",
  navy: "text-[#262467]",
  auto: "odg-watermark-auto",
};

const CORNER_CLASS: Record<Corner, string> = {
  "top-left": "items-start justify-start",
  "top-right": "items-start justify-end",
  "bottom-left": "items-end justify-start",
  "bottom-right": "items-end justify-end",
};

const FONT: CSSProperties = {
  fontFamily: "var(--font-geist-sans), 'Segoe UI', Arial, sans-serif",
};

export function Watermark({
  variant = "corner",
  tone = "auto",
  opacity,
  position = "bottom-right",
  label = true,
  fixed = false,
  className,
  style,
}: WatermarkProps) {
  const layer = cn(
    fixed ? "fixed" : "absolute",
    "inset-0 z-10 pointer-events-none select-none overflow-hidden",
    TONE_CLASS[tone],
    className,
  );
  const layerStyle: CSSProperties = {
    opacity: opacity ?? DEFAULT_OPACITY[variant],
    ...style,
  };

  if (variant === "tile") {
    return (
      <div aria-hidden className={layer} style={layerStyle}>
        <TilePattern />
      </div>
    );
  }

  if (variant === "center") {
    return (
      <div
        aria-hidden
        className={cn(layer, "flex items-center justify-center p-6")}
        style={layerStyle}
      >
        <div className="flex w-[min(60%,26rem)] flex-col items-center gap-[6%]">
          <OceanMark className="w-[62%]" />
          {label && (
            <span
              className="whitespace-nowrap text-center leading-none text-[clamp(1.1rem,4vw,2.6rem)]"
              style={FONT}
            >
              <span className="font-bold">Ocean</span> Data Group
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      aria-hidden
      className={cn(layer, "flex p-4 sm:p-6", CORNER_CLASS[position])}
      style={layerStyle}
    >
      <div className="flex items-center gap-2">
        <OceanMark className="h-7 w-auto sm:h-8" />
        {label && (
          <span className="flex flex-col text-[0.7rem] leading-[1.05] sm:text-xs" style={FONT}>
            <span className="font-bold tracking-wide">Ocean</span>
            <span>Data Group</span>
          </span>
        )}
      </div>
    </div>
  );
}

/* ---------- tiled pattern ---------- */

// Pattern cell (px). Two staggered motifs per cell so rows interleave.
const CELL = { w: 460, h: 170 };
const MARK_H = 34;
const MARK_SCALE = MARK_H / MARK_BOX.h;
const MOTIFS = [
  { x: 0, y: 22 },
  { x: CELL.w / 2, y: 22 + CELL.h / 2 },
];

function TilePattern() {
  const id = `odgt${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  return (
    <svg className="absolute inset-0 h-full w-full">
      <defs>
        <OceanMarkDefs id={id} />
        <pattern
          id={`${id}-tile`}
          width={CELL.w}
          height={CELL.h}
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(-24)"
        >
          {MOTIFS.map(({ x, y }, i) => (
            <g key={i} transform={`translate(${x} ${y})`}>
              <use
                href={`#${id}-shape`}
                transform={`scale(${MARK_SCALE}) translate(${-MARK_BOX.x} ${-MARK_BOX.y})`}
              />
              <text
                x={MARK_H * 1.06 + 9}
                y={MARK_H / 2}
                dominantBaseline="central"
                fill="currentColor"
                fontSize={11}
                fontWeight={600}
                letterSpacing={1.6}
                style={FONT}
              >
                OCEAN DATA GROUP
              </text>
            </g>
          ))}
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill={`url(#${id}-tile)`} />
    </svg>
  );
}


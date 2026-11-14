"use client";

/**
 * Aceternity "SaaS landing page" hero, adapted to StockSentinel: the beam diagram shows the
 * inputs (prices, news, market mood, risk) flowing into StockSentinel and today's picks
 * leaving on the other side. Nav, copy and nodes come from the page; no third-party logos.
 */
import { wrap } from "motion";
import { AnimatePresence, motion, useAnimationFrame, useReducedMotion } from "motion/react";
import { useId, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { OceanMark } from "@/components/brand/ocean-mark";
import { Button, text } from "@/components/saas-landing-page/ui";

type Point = { x: number; y: number };

/** One square on the diagram: an input icon or a ticker. */
export type HeroNode = { key: string; label: string; content: ReactNode; href?: string; muted?: boolean };

type PlacedNode = Point & HeroNode;
type Trace = { id: string; d: string };
type Offshoot = Point & { dx: number; dy: number };
type DiagramLabel = Point & { text: string; variant: "pill" | "note" };

type DiagramLayout = {
  canvas: { w: number; h: number };
  hub: Point;
  traces: Trace[];
  junctions: Point[];
  offshoots: Offshoot[];
  nodes: PlacedNode[];
  labels: DiagramLabel[];
};

export type HeroLabels = { inPill: string; outPill: string; inNote: string; outNote: string };

const WIDE_IN: Point[] = [
  { x: 200, y: 88 },
  { x: 120, y: 200 },
  { x: 264, y: 248 },
  { x: 200, y: 328 },
];
const WIDE_OUT: Point[] = [
  { x: 1000, y: 88 },
  { x: 936, y: 248 },
  { x: 1080, y: 200 },
  { x: 1000, y: 328 },
];
const NARROW_COLUMNS = [60, 140, 220, 300];

function wideLayout(inputs: HeroNode[], outputs: HeroNode[], l: HeroLabels): DiagramLayout {
  return {
    canvas: { w: 1200, h: 400 },
    hub: { x: 600, y: 200 },
    traces: [
      { id: "left-main", d: "M 150 200 H 554" },
      { id: "right-main", d: "M 646 200 H 1050" },
      { id: "upper", d: "M 200 88 H 1000" },
      { id: "hub-riser", d: "M 600 154 V 88" },
      { id: "col-left", d: "M 200 88 V 306" },
      { id: "col-right", d: "M 1000 88 V 306" },
      { id: "in-3", d: "M 264 200 V 232" },
      { id: "out-2", d: "M 936 200 V 232" },
    ],
    junctions: [
      { x: 200, y: 88 },
      { x: 264, y: 200 },
      { x: 200, y: 200 },
      { x: 600, y: 88 },
      { x: 936, y: 200 },
      { x: 1000, y: 200 },
      { x: 1000, y: 88 },
    ],
    offshoots: [
      { x: 320, y: 88, dx: 0, dy: -12 },
      { x: 390, y: 88, dx: 0, dy: 12 },
      { x: 810, y: 88, dx: 0, dy: -12 },
      { x: 880, y: 88, dx: 0, dy: 12 },
      { x: 330, y: 200, dx: 0, dy: -12 },
      { x: 410, y: 200, dx: 0, dy: 12 },
      { x: 790, y: 200, dx: 0, dy: -12 },
      { x: 870, y: 200, dx: 0, dy: 12 },
      { x: 200, y: 140, dx: -12, dy: 0 },
      { x: 200, y: 260, dx: 12, dy: 0 },
      { x: 1000, y: 140, dx: 12, dy: 0 },
      { x: 1000, y: 260, dx: -12, dy: 0 },
    ],
    nodes: [...inputs.slice(0, 4).map((n, i) => ({ ...n, ...WIDE_IN[i] })), ...outputs.slice(0, 4).map((n, i) => ({ ...n, ...WIDE_OUT[i] }))],
    labels: [
      { text: l.inPill, variant: "pill", x: 432, y: 88 },
      { text: l.outPill, variant: "pill", x: 768, y: 88 },
      { text: l.inNote, variant: "note", x: 408, y: 248 },
      { text: l.outNote, variant: "note", x: 792, y: 248 },
    ],
  };
}

function narrowLayout(inputs: HeroNode[], outputs: HeroNode[], l: HeroLabels): DiagramLayout {
  return {
    canvas: { w: 360, h: 440 },
    hub: { x: 180, y: 220 },
    traces: [
      ...NARROW_COLUMNS.map((x) => ({ id: `in-${x}`, d: `M ${x} 48 V 120 H 180 V 190` })),
      ...NARROW_COLUMNS.map((x) => ({ id: `out-${x}`, d: `M 180 250 V 320 H ${x} V 392` })),
    ],
    junctions: [
      { x: 60, y: 120 },
      { x: 300, y: 120 },
      { x: 60, y: 320 },
      { x: 300, y: 320 },
    ],
    offshoots: [
      { x: 100, y: 120, dx: 0, dy: 10 },
      { x: 260, y: 120, dx: 0, dy: -10 },
      { x: 100, y: 320, dx: 0, dy: -10 },
      { x: 260, y: 320, dx: 0, dy: 10 },
    ],
    nodes: [
      ...inputs.slice(0, 4).map((n, i) => ({ ...n, x: NARROW_COLUMNS[i], y: 48 })),
      ...outputs.slice(0, 4).map((n, i) => ({ ...n, x: NARROW_COLUMNS[i], y: 392 })),
    ],
    labels: [
      { text: l.inPill, variant: "pill", x: 180, y: 120 },
      { text: l.outPill, variant: "pill", x: 180, y: 320 },
    ],
  };
}

export type NavLink = { href: string; label: string };

export const ODG_URL = "https://oceandatagroup.ai";

function pct(canvas: DiagramLayout["canvas"], point: Point) {
  return {
    left: `${(point.x / canvas.w) * 100}%`,
    top: `${(point.y / canvas.h) * 100}%`,
  };
}

function Wordmark() {
  return (
    <a
      href="#top"
      aria-label="StockSentinel, back to top"
      className="inline-flex items-center gap-2 rounded-full text-(--folio-ink) focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-(--folio-ink)"
    >
      <OceanMark className="size-6 text-(--folio-ink)" />
      <span className="flex flex-col leading-tight">
        <span className="text-sm font-medium tracking-tight">StockSentinel</span>
        <span className="text-[10px] text-(--folio-muted)">
          Powered by <span className="font-semibold">Ocean Data Group</span>
        </span>
      </span>
    </a>
  );
}

function NavBar({ links, actions }: { links: NavLink[]; actions?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  return (
    <header className="sticky top-0 z-40 border-b border-transparent bg-(--folio-paper)/85 backdrop-blur-md">
      <nav aria-label="Primary" className="mx-auto grid max-w-7xl grid-cols-[1fr_auto] items-center gap-3 px-4 py-3 sm:px-6 lg:grid-cols-[1fr_auto_1fr] lg:px-8">
        <Wordmark />
        <ul role="list" className="hidden items-center gap-7 lg:flex">
          {links.map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                className="text-sm text-(--folio-muted) hover:text-(--folio-ink) focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-(--folio-ink)"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>
        <div className="flex items-center gap-1.5 justify-self-end">
          {actions}
          <button
            type="button"
            className="relative -mr-2 grid size-11 place-items-center rounded-md text-(--folio-ink) focus-visible:outline-2 focus-visible:outline-(--folio-ink) lg:hidden"
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((value) => !value)}
          >
            <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
            <span aria-hidden="true" className="flex w-4 flex-col gap-1">
              <span className={cn("h-px w-full bg-current", open && "translate-y-1 rotate-45")} />
              <span className={cn("h-px w-full bg-current", open && "opacity-0")} />
              <span className={cn("h-px w-full bg-current", open && "-translate-y-1 -rotate-45")} />
            </span>
          </button>
        </div>
      </nav>
      <AnimatePresence>
        {open ? (
          <motion.div
            id="mobile-nav"
            initial={shouldReduceMotion ? false : { opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={shouldReduceMotion ? { opacity: 1 } : { opacity: 0, y: -6 }}
            transition={{ duration: 0.2, ease: [0.23, 1, 0.32, 1] }}
            className="border-t border-black/6 bg-(--folio-paper) px-4 py-4 lg:hidden dark:border-white/8"
          >
            <ul role="list" className="flex flex-col gap-1">
              {links.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    onClick={() => setOpen(false)}
                    className="flex min-h-11 items-center rounded-xl px-3 text-base text-(--folio-ink) focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--folio-ink)"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </header>
  );
}

function ConnectionLayer({ layout }: { layout: DiagramLayout }) {
  const uid = useId();
  const beamRefs = useRef<(SVGPathElement | null)[]>([]);
  const sparkRefs = useRef<(SVGPathElement | null)[]>([]);
  const offsets = useRef<number[]>([]);

  // Beams run even under prefers-reduced-motion (owner's choice, like the headline loop);
  // rendering them unconditionally also keeps server and client markup identical.
  useAnimationFrame((_, delta) => {
    beamRefs.current.forEach((el, index) => {
      if (!el) {
        return;
      }
      const length = el.getTotalLength();
      if (length === 0) {
        return;
      }
      const current = offsets.current[index] ?? 0;
      const next = wrap(0, length, current - (length * delta) / 3600);
      offsets.current[index] = next;
      el.style.strokeDashoffset = `${next}`;

      const spark = sparkRefs.current[index];
      if (spark) {
        spark.style.strokeDashoffset = `${wrap(0, length, next + length * 0.38)}`;
      }
    });
  });

  return (
    <svg
      viewBox={`0 0 ${layout.canvas.w} ${layout.canvas.h}`}
      className="pointer-events-none absolute inset-0 size-full"
      fill="none"
      aria-hidden="true"
    >
      <defs>
        <filter id={`${uid}-glow`} x="-20%" y="-40%" width="140%" height="180%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="3.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {layout.traces.map((trace) => (
        <path
          key={`${trace.id}-glow`}
          d={trace.d}
          className="stroke-(--folio-blue)/20 dark:stroke-(--folio-blue)/30"
          strokeWidth={7}
          strokeLinecap="round"
        />
      ))}

      {layout.traces.map((trace) => (
        <path
          key={`${trace.id}-base`}
          d={trace.d}
          className="stroke-(--folio-line)"
          strokeWidth={1.25}
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
      ))}

      {layout.offshoots.map((tick, index) => (
        <path
          key={`tick-${index}`}
          d={`M ${tick.x} ${tick.y} l ${tick.dx} ${tick.dy}`}
          className="stroke-(--folio-line)"
          strokeWidth={1.15}
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
      ))}

      {layout.junctions.map((pad, index) => (
        <circle
          key={`pad-${index}`}
          cx={pad.x}
          cy={pad.y}
          r={3}
          className="fill-(--folio-paper) stroke-(--folio-line)"
          strokeWidth={1.15}
        />
      ))}

      {layout.traces.map((trace, index) => (
            <g key={`${trace.id}-beams`}>
              <path
                ref={(node) => {
                  beamRefs.current[index] = node;
                }}
                d={trace.d}
                className="stroke-[#4b6bfb] dark:stroke-[#8aa0ff]"
                strokeWidth={2}
                strokeLinecap="round"
                strokeDasharray="22 190"
                filter={`url(#${uid}-glow)`}
              />
              <path
                ref={(node) => {
                  sparkRefs.current[index] = node;
                }}
                d={trace.d}
                className="stroke-[#8aa0ff] dark:stroke-[#c5d0ff]"
                strokeWidth={1.35}
                strokeLinecap="round"
                strokeDasharray="7 205"
              />
            </g>
          ))}
    </svg>
  );
}

function DiagramCanvas({ layout, className }: { layout: DiagramLayout; className?: string }) {
  const shouldReduceMotion = useReducedMotion();
  const { canvas } = layout;

  return (
    <div className={cn("relative w-full", className)} style={{ aspectRatio: `${canvas.w} / ${canvas.h}` }}>
      <ConnectionLayer layout={layout} />

      {layout.labels.map((label) =>
        label.variant === "pill" ? (
          <span
            key={label.text}
            className="absolute z-10 inline-flex -translate-x-1/2 -translate-y-1/2 rounded-full bg-(--folio-card) px-2.5 py-1 text-[0.6875rem] whitespace-nowrap text-(--folio-muted) ring-1 ring-black/8 dark:ring-white/10"
            style={pct(canvas, label)}
          >
            {label.text}
          </span>
        ) : (
          <p key={label.text} className="absolute -translate-x-1/2 -translate-y-1/2 text-[0.6875rem] whitespace-nowrap text-(--folio-muted)" style={pct(canvas, label)}>
            {label.text}
          </p>
        ),
      )}

      <div className="absolute z-20 -translate-x-1/2 -translate-y-1/2" style={pct(canvas, layout.hub)}>
        <a
          href={ODG_URL}
          target="_blank"
          rel="noopener"
          aria-label="Ocean Data Group (opens in a new tab)"
          className="grid size-14 place-items-center rounded-2xl bg-linear-to-b from-[#6d8bff] to-(--folio-blue) text-white shadow-[0_8px_20px_-8px_#4b6bfb] transition-transform hover:scale-105 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-(--folio-ink) sm:size-16 dark:shadow-none"
        >
          <OceanMark className="size-8 sm:size-9" />
        </a>
      </div>

      {layout.nodes.map((node) => {
        const Tag = node.href ? "a" : "span";
        return (
          <Tag
            key={node.key}
            {...(node.href ? { href: node.href } : {})}
            aria-label={node.label}
            title={node.label}
            className="absolute z-20 -translate-x-1/2 -translate-y-1/2 rounded-xl focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--folio-ink)"
            style={pct(canvas, node)}
          >
            <motion.span
              whileHover={shouldReduceMotion ? undefined : { y: -3 }}
              transition={{ duration: 0.2, ease: [0.23, 1, 0.32, 1] }}
              className={cn(
                "grid size-10 place-items-center rounded-xl bg-(--folio-card) text-[0.6875rem] font-semibold tracking-tight text-(--folio-ink) shadow-[0_8px_18px_-10px_oklch(0_0_0/0.28)] ring-1 ring-black/6 sm:size-11 dark:shadow-none dark:inset-ring dark:ring-white/10 dark:inset-ring-white/6",
                node.muted && "opacity-55",
              )}
            >
              {node.content}
            </motion.span>
          </Tag>
        );
      })}
    </div>
  );
}

export function Hero({
  links,
  actions,
  eyebrow,
  title,
  lead,
  primary,
  secondary,
  extra,
  inputs,
  outputs,
  labels,
  caption,
}: {
  links: NavLink[];
  actions?: ReactNode;
  eyebrow?: ReactNode;
  title: ReactNode;
  lead: ReactNode;
  primary: { href: string; label: string };
  secondary: { href: string; label: string };
  extra?: ReactNode;
  inputs: HeroNode[];
  outputs: HeroNode[];
  labels: HeroLabels;
  caption: string;
}) {
  return (
    <div id="top" className="relative isolate w-full overflow-x-clip bg-(--folio-paper) text-(--folio-ink) antialiased">
      <NavBar links={links} actions={actions} />

      <section aria-labelledby="hero-heading" className="relative px-4 pt-14 sm:px-6 sm:pt-20 lg:px-8">
        <div className="flex flex-col items-center gap-5 sm:gap-6">
          {eyebrow}
          <h1 id="hero-heading" className="text-center text-4xl font-medium tracking-tight text-balance sm:text-5xl lg:text-6xl">
            {title}
          </h1>
          <p className={cn(text.lead, "max-w-[46ch] text-center")}>{lead}</p>
          <div className="flex flex-wrap items-center justify-center gap-2.5">
            <Button href={primary.href} variant="solid">
              {primary.label}
            </Button>
            <Button href={secondary.href} variant="outline">
              {secondary.label}
            </Button>
          </div>
          {extra}
        </div>
      </section>

      <section aria-label={caption} className="pb-12 md:pb-20">
        <figure className="relative mx-auto mt-10 w-full max-w-6xl px-2 sm:px-4">
          <DiagramCanvas layout={narrowLayout(inputs, outputs, labels)} className="mx-auto max-w-sm md:hidden" />
          <DiagramCanvas layout={wideLayout(inputs, outputs, labels)} className="max-md:hidden" />
          <figcaption className="sr-only">{caption}</figcaption>
        </figure>
      </section>
    </div>
  );
}

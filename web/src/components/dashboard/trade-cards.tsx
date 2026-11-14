"use client";

/**
 * Today's trades as cards; a click opens the asset spotlight.
 * Spotlight = Aceternity "Expandable Card" (shared layoutId morph, outside click / Esc closes).
 */
import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "motion/react";
import { useLenis } from "lenis/react";
import { IconArrowDownRight, IconArrowUpRight, IconX } from "@tabler/icons-react";
import { useOutsideClick } from "@/hooks/use-outside-click";
import { cn } from "@/lib/utils";
import { num, pct, signedPct, shortDate, SECTOR_EN, STRATEGY, REGIME } from "@/lib/format";
import type { News, Prices, Signal } from "@/lib/data";
import { AssetBody, priceChange } from "./asset-detail";
import { Pill } from "./panel";
import { Locked, usePaywall } from "./paywall";

/** Custom: a small P(win) ring (no ring primitive in the registry). */
function WinRing({ p, size = 56 }: { p: number | null; size?: number }) {
  const r = size / 2 - 5;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, p ?? 0));
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Probability of reaching take-profit ${pct(p, 0)}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={5} className="stroke-neutral-200 dark:stroke-white/[0.08]" />
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        strokeWidth={5}
        strokeLinecap="round"
        strokeDasharray={c}
        initial={{ strokeDashoffset: c }}
        whileInView={{ strokeDashoffset: c * (1 - v) }}
        viewport={{ once: true }}
        transition={{ duration: 1.1, ease: "easeOut" }}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        className="stroke-cyan-500"
      />
      <text x="50%" y="50%" dominantBaseline="central" textAnchor="middle" className="fill-current text-[13px] font-semibold tabular-nums">
        {pct(p, 0)}
      </text>
    </svg>
  );
}

function TradeCard({ s, id, onOpen, fresh }: { s: Signal; id: string; onOpen: () => void; fresh: boolean }) {
  const long = s.side === 1;
  const { locked } = usePaywall();
  // Locked: placeholders only (the public data has no ticker, levels, odds or size).
  return (
    <motion.button
      layoutId={`card-${s.ticker}-${id}`}
      onClick={onOpen}
      whileHover={{ y: -4 }}
      className="group relative flex w-full flex-col gap-4 rounded-2xl border border-neutral-200 bg-white p-5 text-left backdrop-blur-sm transition-colors hover:border-cyan-500/50 focus-visible:ring-2 focus-visible:ring-cyan-500 focus-visible:outline-none dark:border-white/[0.1] dark:bg-[#1c1c1a]"
    >
      {fresh && <span className="absolute -top-2 right-4 rounded-full bg-cyan-500 px-2 py-0.5 text-[10px] font-bold tracking-wider text-white uppercase">New</span>}
      <div className="flex items-start justify-between gap-3">
        <div>
          <motion.h3 layoutId={`title-${s.ticker}-${id}`} className="text-2xl font-bold tracking-tight">
            <Locked badge>{locked ? "XXXX" : s.ticker}</Locked>
          </motion.h3>
          <p className="text-xs text-neutral-500">
            <Locked>{locked ? "Sector name" : SECTOR_EN[s.sector ?? ""] ?? s.sector}</Locked> · {STRATEGY[s.strategy]}
          </p>
        </div>
        <Locked>
          <WinRing p={locked ? 0.5 : s.p_success} />
        </Locked>
      </div>
      <div className="flex items-center gap-2">
        <span className={cn("inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-sm font-semibold", long ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300" : "bg-rose-500/15 text-rose-700 dark:text-rose-300")}>
          {long ? <IconArrowUpRight className="h-4 w-4" /> : <IconArrowDownRight className="h-4 w-4" />}
          {long ? "Long" : "Short"}
        </span>
        <span className="text-sm text-neutral-600 dark:text-neutral-300">
          size <Locked>{locked ? "0.0%" : pct(s.kelly_size)}</Locked>
        </span>
      </div>
      <dl className="grid grid-cols-3 gap-2 text-xs">
        {[
          ["Target", locked ? "000.00" : num(s.tp), "text-emerald-600 dark:text-emerald-400", true],
          ["Stop", locked ? "000.00" : num(s.sl), "text-rose-600 dark:text-rose-400", true],
          ["Exit by", shortDate(s.t_max_date), "", false],
        ].map(([k, v, c, gated]) => (
          <div key={String(k)} className="rounded-lg bg-neutral-100 px-2 py-1.5 dark:bg-white/[0.04]">
            <dt className="text-neutral-500">{k}</dt>
            <dd className={cn("font-semibold tabular-nums", String(c))}>{gated ? <Locked>{v}</Locked> : v}</dd>
          </div>
        ))}
      </dl>
      <span className="text-xs font-medium text-cyan-700 group-hover:underline dark:text-cyan-300">{locked ? "Unlock to see this pick →" : "Open details →"}</span>
    </motion.button>
  );
}

export function TradeCards({ signals, prices, news, newTickers }: { signals: Signal[]; prices: Prices | null; news: News | null; newTickers: string[] }) {
  const [active, setActiveRaw] = useState<Signal | null>(null);
  const { locked, openUnlock } = usePaywall();
  // Locked: a card opens the unlock dialog, never the spotlight (it needs the real ticker).
  const setActive = (s: Signal | null) => (s && locked ? openUnlock() : setActiveRaw(s));
  const ref = useRef<HTMLDivElement>(null);
  const id = useId();
  const lenis = useLenis();
  // Portal: the Reveal wrapper's transform would trap a fixed modal under the navbar.
  const [portal, setPortal] = useState(false);
  useEffect(() => setPortal(true), []);

  const trades = signals.filter((s) => s.kelly_size > 0).sort((a, b) => b.kelly_size - a.kelly_size);
  const idle = signals.filter((s) => !(s.kelly_size > 0)).sort((a, b) => a.ticker.localeCompare(b.ticker));
  const stress = signals[0]?.market_regime === "stress";

  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setActive(null);
    document.body.style.overflow = "hidden";
    lenis?.stop();
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = "";
      lenis?.start();
      window.removeEventListener("keydown", onKey);
    };
  }, [active, lenis]);

  useOutsideClick(ref, () => setActive(null));

  const series = active ? prices?.series[active.ticker] ?? [] : [];
  const change = priceChange(series);
  const articles = active ? (news?.articles ?? []).filter((a) => a.tickers?.includes(active.ticker)).sort((a, b) => b.published_at.localeCompare(a.published_at)) : [];

  return (
    <div>

      {trades.length ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {trades.map((s) => (
            <TradeCard key={s.ticker} s={s} id={id} onOpen={() => setActive(s)} fresh={newTickers.includes(s.ticker)} />
          ))}
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-neutral-300 p-8 text-center text-sm text-neutral-600 dark:border-white/15 dark:text-neutral-300">
          {stress ? "The market is in a stress period, and StockSentinel does not trade in stress periods." : "No asset met StockSentinel's rules today."}
        </div>
      )}

      {idle.length > 0 && locked && (
        <p className="mt-6 text-xs text-neutral-500">
          {idle.length} other assets have no pick today. <button type="button" onClick={openUnlock} className="font-medium text-cyan-700 underline-offset-4 hover:underline dark:text-cyan-300">Unlock</button> to see their names and why.
        </p>
      )}
      {idle.length > 0 && !locked && (
        <div className="mt-6">
          <p className="mb-2 text-xs text-neutral-500">Watching ({idle.length}): no trade today. Tap one to see why.</p>
          <div className="flex flex-wrap gap-1.5">
            {idle.map((s) => (
              <motion.button
                key={s.ticker}
                layoutId={`card-${s.ticker}-${id}`}
                onClick={() => setActive(s)}
                className="rounded-full border border-neutral-200 bg-white/60 px-3 py-1 text-xs font-medium text-neutral-600 transition-colors hover:border-cyan-500/50 hover:text-neutral-900 focus-visible:ring-2 focus-visible:ring-cyan-500 focus-visible:outline-none dark:border-white/[0.08] dark:bg-white/[0.03] dark:text-neutral-400 dark:hover:text-white"
              >
                <motion.span layoutId={`title-${s.ticker}-${id}`}>{s.ticker}</motion.span>
              </motion.button>
            ))}
          </div>
        </div>
      )}

      {portal && createPortal(<>
      <AnimatePresence>
        {active && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[990] bg-black/40 backdrop-blur-sm" />}
      </AnimatePresence>
      <AnimatePresence>
        {active && (
          <div className="fixed inset-0 z-[1000] grid place-items-center sm:p-6" role="dialog" aria-modal="true" aria-labelledby={`title-${active.ticker}`}>
            <motion.div
              layoutId={`card-${active.ticker}-${id}`}
              ref={ref}
              data-lenis-prevent
              className="h-full w-full max-w-3xl overflow-x-hidden overflow-y-auto border border-neutral-200 bg-white p-5 sm:h-auto sm:max-h-[92vh] sm:rounded-3xl md:p-7 dark:border-white/[0.1] dark:bg-[#1c1c1a]"
            >
              <div className="mb-4 flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs text-neutral-500">{SECTOR_EN[active.sector ?? ""] ?? active.sector} · regime {REGIME[active.regime ?? ""] ?? "—"}</p>
                  <motion.h3 layoutId={`title-${active.ticker}-${id}`} id={`title-${active.ticker}`} className="text-3xl font-bold tracking-tight">
                    {active.ticker}
                  </motion.h3>
                  <div className="mt-1 flex flex-wrap items-center gap-2">
                    <span className="text-xl font-semibold tabular-nums">{num(active.last_close)}</span>
                    <span className="text-xs text-neutral-500">USD</span>
                    {change != null && <Pill tone={change >= 0 ? "good" : "bad"}>{signedPct(change)} · 60 sessions</Pill>}
                  </div>
                </div>
                <button onClick={() => setActive(null)} aria-label="Close" className="rounded-full bg-neutral-100 p-2 hover:bg-neutral-200 dark:bg-white/[0.06] dark:hover:bg-white/[0.12]">
                  <IconX className="h-4 w-4" />
                </button>
              </div>
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.15 } }} exit={{ opacity: 0 }}>
                <AssetBody signal={active} series={series} articles={articles} />
              </motion.div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
      </>, document.body)}
    </div>
  );
}

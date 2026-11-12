"use client";

/** Custom: the registry has no data table; styled to match the Aceternity surfaces. */
import { cn } from "@/lib/utils";
import { pct, num, REGIME, SIDE, STRATEGY, SECTOR_EN, shortDate } from "@/lib/format";
import type { Signal } from "@/lib/data";
import { Panel, Pill } from "./panel";
import { Tooltip } from "@/components/ui/tooltip-card";


// First column stays put while the table scrolls sideways on phones (opaque = panel color).
const STICKY = "sticky left-0 z-[1] bg-white sm:static sm:bg-transparent dark:bg-[#14152b] sm:dark:bg-transparent";
export function SignalsBoard({ signals, selected, onSelect }: { signals: Signal[]; selected: string | null; onSelect: (t: string) => void }) {
  const rows = [...signals].sort((a, b) => (b.kelly_size || 0) - (a.kelly_size || 0) || a.ticker.localeCompare(b.ticker));
  const active = rows.filter((s) => s.kelly_size > 0).length;
  const asOf = signals[0]?.as_of;
  return (
    <Panel
      id="signals"
      eyebrow="Today"
      title="Signals board"
      aside={
        <div className="flex flex-wrap gap-2">
          <Pill tone="info">{active} active of {signals.length}</Pill>
          {asOf && <Pill>as of {shortDate(asOf)}</Pill>}
        </div>
      }
    >
      <p className="mb-2 text-xs text-neutral-500 sm:hidden">Swipe the table sideways for levels and dates →</p>
      <div className="-mx-2 overflow-x-auto overscroll-x-contain">
        <table className="w-full min-w-[640px] border-separate border-spacing-0 text-sm">
          <thead>
            <tr className="text-left text-[11px] tracking-wide text-neutral-500 uppercase">
              {["Asset", "Regime", "Setup", "Side", "P(win)", "Size", "Take profit", "Stop", "Exit by"].map((h) => (
                <th key={h} className={cn("border-b border-neutral-200 px-2 py-2 font-medium dark:border-white/[0.08]", h === "Asset" && STICKY)}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => {
              const on = s.kelly_size > 0;
              return (
                <tr
                  key={s.ticker}
                  onClick={() => onSelect(s.ticker)}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(s.ticker)}
                  tabIndex={0}
                  aria-selected={selected === s.ticker}
                  className={cn(
                    "cursor-pointer transition-colors hover:bg-neutral-100 focus-visible:bg-neutral-100 focus-visible:outline-none dark:hover:bg-white/[0.04] dark:focus-visible:bg-white/[0.06]",
                    selected === s.ticker && "bg-cyan-500/[0.07] dark:bg-cyan-400/[0.07]",
                    !on && "text-neutral-500",
                  )}
                >
                  <td className={cn("border-b border-neutral-100 px-2 py-2.5 dark:border-white/[0.04]", STICKY)}>
                    <span className="font-semibold text-neutral-900 dark:text-white">{s.ticker}</span>
                    <span className="ml-2 hidden text-xs text-neutral-500 sm:inline">{SECTOR_EN[s.sector ?? ""] ?? s.sector}</span>
                  </td>
                  <td className="border-b border-neutral-100 px-2 py-2.5 dark:border-white/[0.04]">
                    <Pill tone={s.regime === "stress" ? "bad" : s.regime === "calm" ? "good" : "neutral"}>{REGIME[s.regime ?? ""] ?? "—"}</Pill>
                  </td>
                  <td className="border-b border-neutral-100 px-2 py-2.5 dark:border-white/[0.04]">{STRATEGY[s.strategy]}</td>
                  <td className="border-b border-neutral-100 px-2 py-2.5 dark:border-white/[0.04]">
                    <span className={cn(s.side === 1 && on && "text-emerald-600 dark:text-emerald-400", s.side === -1 && on && "text-rose-600 dark:text-rose-400")}>
                      {SIDE[String(s.side)]}
                    </span>
                  </td>
                  <td className="border-b border-neutral-100 px-2 py-2.5 tabular-nums dark:border-white/[0.04]">{pct(s.p_success, 0)}</td>
                  <td className="border-b border-neutral-100 px-2 py-2.5 tabular-nums dark:border-white/[0.04]">
                    {on ? (
                      s.size_limit === "es" ? (
                        <Tooltip content="Capped by the Expected Shortfall limit: a 1-in-100 tail over this trade's life costs at most 1% of equity.">
                          <span className="underline decoration-dotted underline-offset-4">{pct(s.kelly_size)}</span>
                        </Tooltip>
                      ) : (
                        pct(s.kelly_size)
                      )
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="border-b border-neutral-100 px-2 py-2.5 tabular-nums dark:border-white/[0.04]">{s.tp ? num(s.tp) : "—"}</td>
                  <td className="border-b border-neutral-100 px-2 py-2.5 tabular-nums dark:border-white/[0.04]">{s.sl ? num(s.sl) : "—"}</td>
                  <td className="border-b border-neutral-100 px-2 py-2.5 dark:border-white/[0.04]">{s.t_max_date ? shortDate(s.t_max_date) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-xs text-neutral-500">
        The market regime (hidden Markov model on SPY) decides whether to trade at all. The primary rule proposes a side; the meta-model keeps it only if
        P(success) beats Kelly, and sizes it at ½ Kelly × volatility target, capped by Expected Shortfall. Select a row for details.
      </p>
    </Panel>
  );
}

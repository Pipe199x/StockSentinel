"use client";

/**
 * "$486,238,450 following StockSentinel since 2019" — the evidence as a scroll story (Aceternity Timeline).
 * Facts only: every sentence is a published number on a fixed model account (ACCOUNT); no verdicts and no
 * index comparison.
 */
import { motion } from "motion/react";
import { Timeline, type TimelineEntry } from "@/components/ui/timeline";
import { LineChart } from "@/components/charts/line-chart";
import { cn } from "@/lib/utils";
import { fin, longDate, num } from "@/lib/format";
import type { ValidationReport } from "@/lib/data";

/** Every figure on the page is for this one account. */
export const ACCOUNT = 486_238_450;

/** "$486,238,450" — whole dollars. */
export const money = (v: unknown) => (fin(v) ? `$${Math.round(Math.abs(v)).toLocaleString("en-US")}` : "—");
/** Chart labels: "$540.2M", "$1.3B", "$2,730" — full digits would overflow the axis gutter. */
export const compactMoney = (v: unknown) =>
  fin(v) ? (Math.abs(v) >= 1e6 ? `$${new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v)}` : money(v)) : "—";
/** "+$164" / "−$4,224". */
export const signedMoney = (v: unknown) => (fin(v) ? `${v < 0 ? "−" : "+"}${money(v)}` : "—");
const percent = (v: unknown, d = 0) => (fin(v) ? `${(v * 100).toFixed(d)}%` : "—");
const monthYear = (iso?: string | null) =>
  iso ? new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(iso.slice(0, 10) + "T00:00:00Z")) : "";

function Big({ children, tone }: { children: React.ReactNode; tone?: "up" | "down" }) {
  return <span className={cn("font-bold tabular-nums", tone === "up" && "text-emerald-600 dark:text-emerald-400", tone === "down" && "text-rose-600 dark:text-rose-400")}>{children}</span>;
}
const toneOf = (v: number | null | undefined) => (fin(v) ? (v > 0 ? "up" : v < 0 ? "down" : undefined) : undefined);

function P({ children }: { children: React.ReactNode }) {
  return <p className="max-w-xl text-base leading-relaxed text-neutral-700 md:text-lg dark:text-neutral-200">{children}</p>;
}

/** Custom: 100 squares, `wins` of them lit, filling in as they scroll into view. */
function HundredGrid({ wins }: { wins: number }) {
  return (
    <div className="mt-4 grid w-fit grid-cols-10 gap-1" role="img" aria-label={`${wins} wins out of 100 trades`}>
      {Array.from({ length: 100 }, (_, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, scale: 0.4 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.008 }}
          className={cn("h-3.5 w-3.5 rounded-[3px] sm:h-4 sm:w-4", i < wins ? "bg-cyan-500" : "bg-neutral-300 dark:bg-white/[0.12]")}
        />
      ))}
    </div>
  );
}

/** Custom: a bar from 0 to 100% with a tick at the reference line. */
function OddsBar({ value, line }: { value: number; line: number }) {
  return (
    <div className="mt-4 max-w-md">
      <div
        role="progressbar"
        aria-label="Chance the results come from skill rather than luck"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
        className="relative h-2.5 rounded-full bg-neutral-200 dark:bg-white/[0.08]"
      >
        <motion.div className="h-full rounded-full bg-cyan-500" initial={{ width: 0 }} whileInView={{ width: `${value * 100}%` }} viewport={{ once: true }} transition={{ duration: 1.2, ease: "easeOut" }} />
        <span aria-hidden className="absolute -top-1.5 h-5.5 w-0.5 rounded-full bg-neutral-800 dark:bg-white" style={{ left: `${line * 100}%` }} />
      </div>
      <div className="relative mt-1.5 h-4 text-xs text-neutral-500">
        <span className="absolute left-0">0%</span>
        <span className="absolute -translate-x-1/2 font-medium text-neutral-700 dark:text-neutral-300" style={{ left: `${line * 100}%` }}>{percent(line)} line</span>
      </div>
    </div>
  );
}

function Tiles({ items }: { items: [string, string, ("up" | "down" | undefined)?][] }) {
  return (
    <div className="mt-4 grid max-w-xl grid-cols-1 gap-2 sm:grid-cols-3">
      {items.map(([k, v, tone]) => (
        <div key={k} className="rounded-xl border border-neutral-200 bg-white/60 p-3 dark:border-white/[0.08] dark:bg-white/[0.03]">
          <p className="text-xs text-neutral-500">{k}</p>
          <p className="mt-1 text-xl"><Big tone={tone}>{v}</Big></p>
        </div>
      ))}
    </div>
  );
}

/** Largest fall from a previous high, in dollars, on the weekly curve. */
function worstDrop(points: { value: number }[]) {
  let peak = -Infinity;
  let worst = 0;
  for (const p of points) {
    peak = Math.max(peak, p.value);
    worst = Math.max(worst, peak - p.value);
  }
  return worst;
}

export function Story({ report }: { report: ValidationReport | null }) {
  const m = report?.meta;
  const chapters: TimelineEntry[] = [];

  if (report && m) {
    const from = report.folds?.[0]?.test_from;
    const wins = Math.round((m.win_rate ?? 0) * 100);
    chapters.push({
      title: monthYear(from) || "The start",
      content: (
        <P>
          <Big>{money(ACCOUNT)}</Big> starts following StockSentinel&apos;s picks{from ? <> on {longDate(from)}</> : ""}. The test runs on <Big>{num(m.years, 1)} years</Big> of market history
          StockSentinel had never seen.
        </P>
      ),
    });
    chapters.push({
      title: `${m.n_trades.toLocaleString()} trades`,
      content: (
        <>
          <P>
            Over those years StockSentinel made <Big>{m.n_trades.toLocaleString()}</Big> trades. Out of every 100, it won <Big>{wins}</Big> and lost <Big>{100 - wins}</Big>.
          </P>
          <HundredGrid wins={wins} />
        </>
      ),
    });

    const cal = m.calendar;
    const curve = (m.equity_curve_weekly ?? []).map((p) => ({ date: p.date, value: ACCOUNT * p.equity }));
    if (cal && fin(cal.total_return)) {
      const end = ACCOUNT * (1 + cal.total_return);
      chapters.push({
        title: monthYear(report.as_of),
        content: (
          <>
            <P>
              Today that account is worth <Big tone={toneOf(cal.total_return)}>{money(end)}</Big> ({signedMoney(end - ACCOUNT)}).
            </P>
            <Tiles
              items={[
                ["Worst drop from a high", curve.length > 1 ? signedMoney(-worstDrop(curve)) : "—", "down"],
                ["Days with money in the market", percent(cal.time_in_market)],
                ["Share of the account in trades, on average", percent(cal.avg_gross_exposure, 1)],
              ]}
            />
            {curve.length > 1 && <LineChart label={`Value of the ${money(ACCOUNT)} account following StockSentinel, weekly`} points={curve} format={compactMoney} area className="mt-4 max-w-2xl" />}
          </>
        ),
      });
    } else {
      chapters.push({
        title: "Today",
        content: <P>The account&apos;s value over time appears here after the next data update.</P>,
      });
    }

    const crashes = Object.entries(report.stress_windows ?? {}).filter(([, w]) => w.covered);
    if (crashes.length) {
      chapters.push({
        title: "Crashes",
        content: (
          <div className="space-y-3">
            {crashes.map(([name, w]) => (
              <P key={name}>
                <span className="font-semibold">{name}</span> ({monthYear(w.from)} to {monthYear(w.to)}):{" "}
                {w.n_trades ? (
                  <>
                    StockSentinel made {w.n_trades} trade{w.n_trades > 1 ? "s" : ""}, and the account moved about <Big tone={toneOf(w.pnl_sum)}>{signedMoney(ACCOUNT * w.pnl_sum)}</Big>.
                  </>
                ) : (
                  <>StockSentinel made no trades, and the account did not move.</>
                )}
              </P>
            ))}
          </div>
        ),
      });
    }

    if (fin(m.dsr)) {
      chapters.push({
        title: "Skill or luck?",
        content: (
          <>
            <P>
              Chance these results come from skill rather than luck: <Big>{percent(m.dsr)}</Big>. Researchers use <Big>{percent(report.dsr_threshold)}</Big> as the line for
              calling a result real. This figure already accounts for the {report.n_trials} versions of StockSentinel that were tried.
            </P>
            <OddsBar value={m.dsr} line={report.dsr_threshold} />
          </>
        ),
      });
    }
  }

  return (
    <div>
      <Timeline data={chapters} />
      <p className="mt-2 text-xs text-neutral-500">Model portfolio; results are hypothetical. Not advice.</p>
    </div>
  );
}

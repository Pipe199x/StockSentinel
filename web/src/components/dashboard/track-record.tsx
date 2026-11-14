"use client";

import { motion } from "motion/react";
import { LineChart } from "@/components/charts/line-chart";
import { cn } from "@/lib/utils";
import { longDate, num, pct, shortDate, SIDE, fin } from "@/lib/format";
import type { PaperRow, PaperSummary, ValidationReport } from "@/lib/data";
import type { VisitDiff } from "@/lib/last-visit";
import { Pill } from "./panel";
import { ACCOUNT, compactMoney, money, signedMoney } from "./story";
import { Locked, usePaywall } from "./paywall";

const REASON: Record<string, string> = { tp: "Take profit", sl: "Stop", time: "Time limit", entry_gap: "Gap at entry", late_publish: "Published after the open" };

/** Custom: progress ring toward Gate 0, with a one-shot spark burst on new take-profits. */
function GateRing({ done, need, burst }: { done: number; need: number; burst: boolean }) {
  const size = 148;
  const r = 62;
  const c = 2 * Math.PI * r;
  const v = Math.min(1, done / Math.max(need, 1));
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={10} className="stroke-neutral-200 dark:stroke-white/[0.07]" />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          strokeWidth={10}
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          whileInView={{ strokeDashoffset: c * (1 - v) }}
          viewport={{ once: true }}
          transition={{ duration: 1.4, ease: "easeOut" }}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          className="stroke-cyan-500"
        />
      </svg>
      <div className="absolute text-center">
        <p className="text-3xl font-bold tabular-nums">{done}<span className="text-base text-neutral-500">/{need}</span></p>
        <p className="text-[11px] text-neutral-500">closed trades</p>
      </div>
      {burst &&
        Array.from({ length: 12 }, (_, i) => (
          <span key={i} aria-hidden className="tp-spark absolute h-2 w-2 rounded-full bg-emerald-400" style={{ ["--a" as string]: `${i * 30}deg`, animationDelay: `${(i % 3) * 60}ms` }} />
        ))}
    </div>
  );
}

/** Consecutive wins (or losses) counting back from the most recent closed trade. */
function streak(closed: PaperRow[]) {
  const rows = [...closed].sort((a, b) => (b.exit_date ?? "").localeCompare(a.exit_date ?? "")).filter((r) => fin(r.ret_net) && r.ret_net !== 0);
  if (!rows.length) return { n: 0, win: true };
  const win = rows[0].ret_net! > 0;
  let n = 0;
  for (const r of rows) {
    if (r.ret_net! > 0 !== win) break;
    n++;
  }
  return { n, win };
}

function PositionChip({ r, pNow }: { r: PaperRow; pNow?: number | null }) {
  const { locked } = usePaywall();
  // Locked: the public ledger row has only side, status and exit date; the rest are placeholders.
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border border-neutral-200 px-3 py-2 text-sm dark:border-white/[0.08]">
      <div className="flex items-center gap-2">
        <span className="font-semibold">
          <Locked>{locked ? "XXXX" : r.ticker}</Locked>
        </span>
        <span className={cn("text-xs", r.side === 1 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400")}>{SIDE[String(r.side)]}</span>
      </div>
      <div className="flex items-center gap-2 text-xs text-neutral-500">
        <span className="tabular-nums text-neutral-700 dark:text-neutral-300">
          <Locked>{locked ? "$00,000,000" : money(ACCOUNT * r.size)}</Locked>
        </span>
        {locked ? (
          <span>
            chance of profit now <Locked>00%</Locked>
          </span>
        ) : (
          fin(pNow) && <span>chance of profit now {pct(pNow, 0)}</span>
        )}
        <span>
          exit by <Locked>{locked ? "Dec 00" : shortDate(r.exit_by)}</Locked>
        </span>
      </div>
    </div>
  );
}

function ClosedRow({ r }: { r: PaperRow }) {
  const up = (r.ret_net ?? 0) > 0;
  return (
    <div className="flex items-center justify-between gap-3 border-t border-neutral-100 py-2 text-sm first:border-0 dark:border-white/[0.05]">
      <span>
        <span className="font-semibold">{r.ticker}</span> <span className="text-xs text-neutral-500">{REASON[r.exit_reason ?? ""] ?? "—"} · {shortDate(r.exit_date)}</span>
      </span>
      <span className={cn("font-semibold tabular-nums", up ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400")}>{signedMoney(ACCOUNT * r.size * (r.ret_net ?? NaN))}</span>
    </div>
  );
}

const monthYear = (iso?: string | null) =>
  iso ? new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(iso.slice(0, 10) + "T00:00:00Z")) : "—";

/** Custom: the line between past and live. Vertical between the cards; horizontal when they stack. */
function PastLiveDivider({ since }: { since: string | null }) {
  return (
    <div role="separator" aria-label={`Past test until ${longDate(since)}, live after`} className="relative flex items-center justify-center py-3 md:px-1 md:py-0">
      <span aria-hidden className="absolute inset-x-0 top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-linear-to-r from-transparent via-[#4b6bfb] to-transparent md:inset-x-auto md:inset-y-0 md:left-1/2 md:top-0 md:h-auto md:w-0.5 md:-translate-x-1/2 md:translate-y-0 md:bg-linear-to-b" />
      <span className="relative z-10 flex flex-col items-center gap-1 rounded-2xl bg-white px-2.5 py-2 text-center ring-1 ring-[#4b6bfb]/40 dark:bg-[#1c1c1a]">
        <span className="text-[10px] font-semibold tracking-wider text-neutral-500 uppercase">Past</span>
        <span aria-hidden className="h-2 w-2 rounded-full bg-[#4b6bfb] shadow-[0_0_0_4px_rgba(75,107,251,0.18)]" />
        <span className="text-[10px] font-semibold tracking-wider text-[#4b6bfb] uppercase">Live</span>
        <span className="text-[10px] whitespace-nowrap text-neutral-500">{shortDate(since)}</span>
      </span>
    </div>
  );
}

/** One model-account result: header, value today with its change, the dollar curve, a footnote. */
function ResultCard({ title, end, points, empty, foot }: { title: string; end: number | null | undefined; points: { date: string; value: number }[]; empty: string; foot: string }) {
  const has = fin(end);
  const value = has ? ACCOUNT * (1 + end) : null;
  return (
    <div className="flex min-w-0 flex-col rounded-2xl bg-[#fafaf7] p-5 ring-1 ring-black/6 dark:bg-[#0e0e0c] dark:ring-white/10">
      <p className="text-xs font-medium tracking-wide text-neutral-500 uppercase">{title}</p>
      <p className="mt-2 flex items-baseline gap-2">
        <span className={cn("text-3xl font-semibold tabular-nums", has && end! > 0 && "text-emerald-600 dark:text-emerald-400", has && end! < 0 && "text-rose-600 dark:text-rose-400")}>
          {has ? money(value) : "—"}
        </span>
        {has && <span className="text-sm text-neutral-500 tabular-nums">{signedMoney(value! - ACCOUNT)}</span>}
      </p>
      <div className="mt-3 flex-1">
        {points.length > 1 ? (
          <LineChart label={`${title}: value of ${money(ACCOUNT)}`} points={points} format={compactMoney} area />
        ) : (
          <div className="grid h-40 place-items-center rounded-xl text-center text-sm text-neutral-500 ring-1 ring-black/5 dark:ring-white/10">{empty}</div>
        )}
      </div>
      {foot && <p className="mt-2 text-xs text-neutral-500">{foot}</p>}
    </div>
  );
}

export function TrackRecord({ paper, visit, report }: { paper: PaperSummary | null; visit: VisitDiff | null; report: ValidationReport | null }) {
  const cal = report?.meta.calendar;
  if (!paper) {
    return (
      <p className="text-sm text-neutral-500">The model portfolio starts with the next data update.</p>
    );
  }
  const g = paper.gate0;
  const done = paper.current_closed ?? paper.counts.closed;
  const st = streak(paper.recent_closed);
  const rescore = Object.fromEntries((paper.rescore ?? []).map((r) => [`${r.ticker}-${r.signal_date}`, r.p_profit_now]));
  const open = paper.positions;
  // Public (locked) file carries only the total share invested; the full one has per-row sizes.
  const invested = Math.min(1, paper.invested ?? open.reduce((s, r) => s + (r.size || 0), 0));
  const value = ACCOUNT * (1 + paper.total_return);
  const tp = visit?.tpHits ?? 0;

  return (
    <div className="relative rounded-3xl bg-white p-5 ring-1 ring-black/8 md:p-8 dark:bg-[#1c1c1a] dark:ring-white/10">
      {tp > 0 && (
        <motion.p initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="mb-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-sm font-medium text-emerald-700 dark:text-emerald-300">
          🎯 {tp} trade{tp > 1 ? "s" : ""} hit take-profit since your last visit.
        </motion.p>
      )}

      {/* Chronological: the same account on past data (left) | the live portfolio (right). */}
      <div className="grid items-stretch gap-4 md:grid-cols-[1fr_auto_1fr] md:gap-6">
        <ResultCard
          title={`Past test · ${monthYear(report?.folds?.[0]?.test_from)} – ${monthYear(report?.as_of)}`}
          end={cal?.total_return}
          points={(report?.meta.equity_curve_weekly ?? []).map((p) => ({ date: p.date, value: ACCOUNT * p.equity }))}
          empty="Appears after the next data update."
          foot={report ? `${report.meta.n_trades.toLocaleString()} trades · ${num(report.meta.years, 1)} years of history` : ""}
        />
        <PastLiveDivider since={paper.started} />
        <ResultCard
          title={`Live · since ${longDate(paper.started)}`}
          end={paper.total_return}
          points={paper.equity_curve.map((p) => ({ date: p.date, value: ACCOUNT * p.equity }))}
          empty="The value chart starts after the second market day."
          foot={`${money(value * invested)} in ${open.length} position${open.length === 1 ? "" : "s"} · ${money(value * (1 - invested))} in cash`}
        />
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-6">
        <GateRing done={done} need={g.min_trades} burst={tp > 0} />
        <div className="max-w-sm space-y-2">
          <p className="text-sm text-neutral-600 dark:text-neutral-300">
            {done} of {g.min_trades} live trades finished. After {g.min_trades}, the live portfolio is compared with the past test.
          </p>
          <p className="text-sm">
            <span className="font-semibold">{st.n ? `${st.n}${st.win ? " 🔥" : ""}` : "—"}</span>{" "}
            <span className="text-neutral-500">{st.win ? "win streak" : "loss streak"}</span>
          </p>
        </div>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Open now ({open.length})</h3>
          <div className="space-y-1.5">
            {open.length ? open.map((r, i) => <PositionChip key={r.ticker ? `${r.ticker}-${r.signal_date}` : `locked-${i}`} r={r} pNow={rescore[`${r.ticker}-${r.signal_date}`]} />) : <p className="text-sm text-neutral-500">No open positions.</p>}
          </div>
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold">Recently closed</h3>
          {paper.recent_closed.length ? (
            <div>{[...paper.recent_closed].reverse().slice(0, 6).map((r) => <ClosedRow key={`${r.ticker}-${r.signal_date}`} r={r} />)}</div>
          ) : (
            <p className="text-sm text-neutral-500">No closed trades yet. The first ones land when today&apos;s positions hit a target, a stop or their exit date.</p>
          )}
        </div>
      </div>

      {paper.gate1 && (
        <p className="mt-6 text-xs text-neutral-500">
          To judge the picks with high confidence, about {paper.gate1.trades_required?.toLocaleString() ?? "—"} independent trades are needed; {num(paper.gate1.live_effective_trades, 0)} so far
          {fin(paper.gate1.years_to_go) ? `, about ${num(paper.gate1.years_to_go, 1)} years at the past pace` : ""}.
        </p>
      )}
    </div>
  );
}

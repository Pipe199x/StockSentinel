"use client";

import { LineChart } from "@/components/charts/line-chart";
import { cn } from "@/lib/utils";
import { num, pct, signedPct, shortDate, SIDE, fin } from "@/lib/format";
import type { PaperRow, PaperSummary } from "@/lib/data";
import { HoverEffect } from "@/components/ui/card-hover-effect";
import { Panel, Pill, Stat } from "./panel";

const REASON: Record<string, string> = { tp: "Take profit", sl: "Stop", time: "Time limit", entry_gap: "Gap at entry" };

/** Custom: ledger tables (no table primitive in the registry). */
function Ledger({ rows, closed }: { rows: PaperRow[]; closed?: boolean }) {
  if (!rows.length) return <p className="py-6 text-center text-sm text-neutral-500">{closed ? "No closed trades yet." : "No open or pending positions."}</p>;
  const head = closed ? ["Asset", "Side", "Size", "Entry", "Exit", "Reason", "Net"] : ["Asset", "Side", "Size", "Status", "Entry", "TP / Stop", "Exit by"];
  return (
    <div className="-mx-1 overflow-x-auto">
      <table className="w-full min-w-[520px] text-sm">
        <thead>
          <tr className="text-left text-[11px] tracking-wide text-neutral-500 uppercase">
            {head.map((h) => (
              <th key={h} className="px-1 py-1.5 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.ticker}-${r.signal_date}`} className="border-t border-neutral-100 dark:border-white/[0.05]">
              <td className="px-1 py-2 font-semibold">{r.ticker}</td>
              <td className={cn("px-1 py-2", r.side === 1 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400")}>{SIDE[String(r.side)]}</td>
              <td className="px-1 py-2 tabular-nums">{pct(r.size)}</td>
              {closed ? (
                <>
                  <td className="px-1 py-2 tabular-nums">{num(r.entry_price)} <span className="text-xs text-neutral-500">{shortDate(r.entry_date)}</span></td>
                  <td className="px-1 py-2 tabular-nums">{num(r.exit_price)} <span className="text-xs text-neutral-500">{shortDate(r.exit_date)}</span></td>
                  <td className="px-1 py-2">{REASON[r.exit_reason ?? ""] ?? "—"}</td>
                  <td className={cn("px-1 py-2 tabular-nums", fin(r.ret_net) && (r.ret_net! > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"))}>{signedPct(r.ret_net)}</td>
                </>
              ) : (
                <>
                  <td className="px-1 py-2"><Pill tone={r.status === "open" ? "info" : "neutral"}>{r.status === "open" ? "Open" : "Fills next open"}</Pill></td>
                  <td className="px-1 py-2 tabular-nums">{r.entry_price ? num(r.entry_price) : "—"}</td>
                  <td className="px-1 py-2 tabular-nums">{num(r.tp)} / {num(r.sl)}</td>
                  <td className="px-1 py-2">{shortDate(r.exit_by)}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Paper({ paper }: { paper: PaperSummary | null }) {
  if (!paper) {
    return (
      <Panel id="paper" eyebrow="Gate 0" title="Paper trading">
        <p className="text-sm text-neutral-500">The paper ledger starts with the next pipeline run. Each signal will fill at the next session&apos;s open and be tracked here.</p>
      </Panel>
    );
  }
  const g = paper.gate0;
  const tone = g.status === "pass" ? "good" : g.status === "fail" ? "bad" : "warn";
  const progress = Math.min(1, paper.counts.closed / g.min_trades);
  return (
    <Panel
      id="paper"
      eyebrow="Gate 0 · simulated, no real money"
      title="Paper trading"
      aside={<Pill tone={tone}>Gate 0: {g.status === "collecting" ? "collecting evidence" : g.status}</Pill>}
    >
      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div>
          <LineChart
            label="Paper portfolio equity, compounding, open positions marked to the close"
            points={paper.equity_curve.map((p) => ({ date: p.date, value: p.equity }))}
            format={(v) => signedPct(v - 1, 1)}
            area
          />
          <p className="mt-2 text-xs text-neutral-500">
            Since {shortDate(paper.started)}: signals fill at the next session&apos;s open; exits follow the backtest rules (gaps fill at the open, a day that
            touches both barriers counts as the stop), net of 10 bps round-trip costs.
          </p>
        </div>
        <div className="space-y-4">
          <HoverEffect className="grid grid-cols-2 gap-1">
            <Stat label="Total return" value={signedPct(paper.total_return)} tone={paper.total_return >= 0 ? "good" : "bad"} />
            <Stat label="Closed / open" value={`${paper.counts.closed} / ${paper.counts.open + paper.counts.pending}`} />
            <Stat label="Win rate" value={pct(paper.win_rate, 0)} />
            <Stat label="EV / trade" value={signedPct(paper.expected_value, 2)} />
          </HoverEffect>
          <div className="rounded-xl border border-neutral-200 p-4 dark:border-white/[0.08]">
            <div className="flex items-center justify-between text-xs text-neutral-500">
              <span>Evidence collected</span>
              <span className="tabular-nums">{paper.counts.closed} / {g.min_trades} trades</span>
            </div>
            <div className="mt-2 h-2 rounded-full bg-neutral-200 dark:bg-white/[0.06]">
              <div className="h-2 rounded-full bg-cyan-500" style={{ width: `${progress * 100}%` }} />
            </div>
            <p className="mt-3 text-sm text-neutral-600 dark:text-neutral-300">{g.reason}.</p>
            <p className="mt-1 text-xs text-neutral-500">
              Live Sharpe/trade {num(g.live_sr_per_trade, 3)} vs backtest CPCV range{" "}
              {g.backtest_cpcv_range ? `${num(g.backtest_cpcv_range[0], 3)} to ${num(g.backtest_cpcv_range[1], 3)}` : "—"}.
            </p>
          </div>
        </div>
      </div>
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Open and pending</h3>
          <Ledger rows={paper.positions} />
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold">Recently closed</h3>
          <Ledger rows={[...paper.recent_closed].reverse()} closed />
        </div>
      </div>
    </Panel>
  );
}

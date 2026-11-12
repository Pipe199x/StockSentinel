"use client";

import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import { Bars } from "@/components/charts/line-chart";
import FrequentlyAskedQuestionsAccordion from "@/block/frequently-asked-questions-accordion";
import { Tooltip } from "@/components/ui/tooltip-card";
import { cn } from "@/lib/utils";
import { num, pct, signedPct, longDate, fin } from "@/lib/format";
import type { ValidationReport } from "@/lib/data";
import { HoverEffect } from "@/components/ui/card-hover-effect";
import { Panel, Pill, Stat } from "./panel";

const HOW_IT_WORKS = [
  { question: "1. Regime first", answer: "A 3-state hidden Markov model on SPY returns and volatility labels the market calm, normal or stress, refitted yearly on past data only. In stress the system stays flat." },
  { question: "2. Primary rule", answer: "Momentum only: when Hurst ≥ 0.5 it follows the 20-day move if price sits on the same side of its fair value. Mean reversion was switched off after it lost money in the replay." },
  { question: "3. Triple barrier", answer: "Take profit and stop are 1.5 × GARCH volatility over the holding period, with a 20-session time limit. The backtest enters at the next open and checks barriers intraday, exactly like the paper ledger." },
  { question: "4. Meta-labeling and size", answer: "XGBoost on stationary features estimates P(profit after costs). Size is ½ Kelly × volatility target, capped so the 1% Expected Shortfall costs at most 1% of equity." },
  { question: "5. Validation", answer: "Purged K-fold with embargo, CPCV and the Deflated Sharpe, which penalizes every configuration ever tried." },
];

export function Validation({ report }: { report: ValidationReport | null }) {
  if (!report) return null;
  const m = report.meta;
  const sig = report.edge_is_significant;
  const windows = Object.entries(report.stress_windows ?? {});
  return (
    <Panel
      id="validation"
      eyebrow="Out-of-sample only"
      title="Honest validation"
      aside={
        <div className="flex flex-wrap gap-2">
          <Pill tone={sig ? "good" : "warn"}>{sig ? "Edge is significant" : "Edge not proven yet"}</Pill>
          <Pill>{report.n_trials} trials logged</Pill>
        </div>
      }
    >
      <BentoGrid className="max-w-none md:auto-rows-auto">
        <BentoGridItem
          className="md:col-span-1 dark:bg-[#101124]"
          title={
            <Tooltip content="Deflated Sharpe Ratio (Bailey & López de Prado): the probability that the Sharpe is real after penalizing every configuration ever tried, skew and fat tails. 0.95 is the bar.">
              <span className="underline decoration-dotted underline-offset-4">Deflated Sharpe</span>
            </Tooltip>
          }
          description={`Bar is ${report.dsr_threshold}. ${sig ? "Above it: the edge is unlikely to be luck." : "Below it: the result could still be luck."}`}
          header={
            <div className="flex items-end gap-3">
              <span className={cn("text-5xl font-bold tabular-nums", sig ? "text-emerald-500" : "text-amber-500")}>{num(m.dsr, 3)}</span>
              <span className="pb-2 text-sm text-neutral-500">/ 1.000</span>
            </div>
          }
        />
        <BentoGridItem
          className="md:col-span-2 dark:bg-[#101124]"
          title="Meta-model trades"
          description={`${num(m.years, 1)} years, ${m.n_trades?.toLocaleString()} trades, costs included. Needs ${m.min_trades_for_edge?.toLocaleString() ?? "—"} to confirm the edge.`}
          header={
            <HoverEffect className="grid grid-cols-2 gap-1 md:grid-cols-4">
              <Stat label="Win rate" value={pct(m.win_rate)} />
              <Stat label="EV / trade" value={signedPct(m.expected_value, 3)} tone={fin(m.expected_value) ? (m.expected_value! > 0 ? "good" : "bad") : undefined} />
              <Stat label="Sharpe (ann.)" value={num(m.sr_annualized, 2)} />
              <Stat label="Worst streak" value={`${m.max_losing_streak ?? "—"}`} sub={`5%-risk drawdown ${pct(m.streak_drawdown_5pct, 0)}`} />
            </HoverEffect>
          }
        />
        <BentoGridItem
          className="md:col-span-2 dark:bg-[#101124]"
          title="CPCV: a distribution of histories, not one"
          description="Sharpe per trade on each combinatorial purged path. One lucky path is not an edge."
          header={<Bars items={report.cpcv_path_sr.map((v, i) => ({ label: `Path ${i + 1}`, value: v }))} format={(v) => num(v, 3)} />}
        />
        <BentoGridItem
          className="md:col-span-1 dark:bg-[#101124]"
          title="Primary rule alone"
          description="Without the meta-model filter. Shows how much of the result comes from the filter."
          header={
            <HoverEffect className="grid grid-cols-2 gap-1">
              <Stat label="Sharpe (ann.)" value={num(report.primary.sr_annualized, 2)} tone={fin(report.primary.sr_annualized) ? (report.primary.sr_annualized! > 0 ? "good" : "bad") : undefined} />
              <Stat label="Trades" value={report.primary.n_trades?.toLocaleString() ?? "—"} />
            </HoverEffect>
          }
        />
        {windows.length > 0 && (
          <BentoGridItem
            className="md:col-span-3 dark:bg-[#101124]"
            title="Crisis windows"
            description="Out-of-sample trades that started inside each crisis, next to what SPY did over the same window."
            header={
              <div className="-mx-1 overflow-x-auto">
                <table className="w-full min-w-[560px] text-sm">
                  <thead>
                    <tr className="text-left text-[11px] tracking-wide text-neutral-500 uppercase">
                      {["Window", "Dates", "SPY", "Trades", "Win rate", "EV / trade", "Summed P&L"].map((h) => (
                        <th key={h} className="px-1 py-1.5 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {windows.map(([name, w]) => (
                      <tr key={name} className="border-t border-neutral-100 dark:border-white/[0.05]">
                        <td className="px-1 py-2 font-medium">{name}</td>
                        <td className="px-1 py-2 text-xs text-neutral-500">{longDate(w.from)} – {longDate(w.to)}</td>
                        <td className="px-1 py-2 tabular-nums">{w.covered ? signedPct(w.benchmark_return, 1) : "no data"}</td>
                        <td className="px-1 py-2 tabular-nums">{w.n_trades}</td>
                        <td className="px-1 py-2 tabular-nums">{pct(w.win_rate, 0)}</td>
                        <td className={cn("px-1 py-2 tabular-nums", fin(w.expected_value) && (w.expected_value! > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"))}>
                          {signedPct(w.expected_value, 2)}
                        </td>
                        <td className="px-1 py-2 tabular-nums">{signedPct(w.pnl_sum, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            }
          />
        )}
        <BentoGridItem
          className="md:col-span-2 dark:bg-[#101124]"
          title="Purged K-fold (one path through time)"
          description="Accuracy and Sharpe per trade on each test fold, with an embargo between train and test."
          header={<Bars items={report.folds.map((f) => ({ label: `${f.test_from.slice(0, 4)}–${f.test_to.slice(0, 4)}`, value: f.sr_per_trade }))} format={(v) => num(v, 3)} />}
        />
        <BentoGridItem
          className="md:col-span-1 dark:bg-[#101124]"
          title="Correlation in crises"
          description="Average correlation between assets. Diversification that disappears in stress is not diversification."
          header={
            <HoverEffect className="grid grid-cols-2 gap-1">
              <Stat label="Calm days" value={num(report.correlation.market_calm_days, 2)} />
              <Stat label="Stress days" value={num(report.correlation.market_stress_days, 2)} tone="bad" />
            </HoverEffect>
          }
        />
      </BentoGrid>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/[0.04] p-4">
          <h3 className="text-sm font-semibold">What this validation does not cover</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-neutral-600 dark:text-neutral-300">
            {report.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </div>
        <div className="rounded-xl border border-neutral-200 p-4 dark:border-white/[0.08]">
          <FrequentlyAskedQuestionsAccordion title="How the strategy works" faqs={HOW_IT_WORKS} />
        </div>
      </div>
    </Panel>
  );
}

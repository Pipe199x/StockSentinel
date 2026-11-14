"use client";

import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import { HoverEffect } from "@/components/ui/card-hover-effect";
import { Tooltip } from "@/components/ui/tooltip-card";
import { num, pct, signedPct, longDate, REGIME, SECTOR_EN, fin } from "@/lib/format";
import type { McStats, RiskReport } from "@/lib/data";
import { Panel, Pill, Stat } from "./panel";

const BUCKET: Record<string, string> = { broad_equity: "Broad equity (SPY/QQQ)", gold: "Gold", rates: "Rates (TLT)", other: "Other" };

function McBlock({ title, m }: { title: string; m?: McStats }) {
  if (!m) return null;
  return (
    <div>
      <p className="mb-1 text-[11px] font-medium tracking-wide text-neutral-500 uppercase">{title}</p>
      <HoverEffect className="grid grid-cols-2 gap-1 md:grid-cols-4">
        <Stat label="P(profit, 1y)" value={pct(m.p_profit, 0)} />
        <Stat label="P(breach limit)" value={pct(m.p_breach_max_loss, 1)} tone={m.p_breach_max_loss > 0.05 ? "bad" : undefined} />
        <Stat label="Drawdown p95" value={pct(m.drawdown_p95, 1)} sub={`p99 ${pct(m.drawdown_p99, 1)}`} />
        <Stat label="1y return 5–95%" value={`${signedPct(m.return_p05, 1)} / ${signedPct(m.return_p95, 1)}`} sub={`CVaR95 ${pct(m.cvar95, 1)}`} />
      </HoverEffect>
    </div>
  );
}

export function Risk({ risk }: { risk: RiskReport | null }) {
  if (!risk) return null;
  const r = risk.regime ?? {};
  const mc = risk.montecarlo;
  const book = risk.book;
  const mkt = risk.garch?.market ?? {};
  const shock = risk.post_shock;
  const dur = r.expected_duration_sessions ?? {};
  return (
    <Panel
      id="risk"
      eyebrow="Informational · never changes a trade"
      title="Risk desk"
      aside={
        <div className="flex flex-wrap gap-2">
          {shock?.active && <Pill tone="warn">Post-shock window: {shock.sessions_since_shock} of {shock.window_sessions} sessions</Pill>}
          {risk.training_mix?.current_regime_rare_in_training && <Pill tone="warn">Regime rare in training</Pill>}
          {(risk.garch_alerts ?? []).length > 0 && <Pill tone="warn">Volatility alerts: {(risk.garch_alerts ?? []).join(", ")}</Pill>}
        </div>
      }
    >
      <BentoGrid className="max-w-none md:auto-rows-auto">
        <BentoGridItem
          className="md:col-span-1 dark:bg-[#1c1c1a]"
          title={
            <Tooltip content="From the market hidden Markov model: expected regime duration is 1/(1 − p_stay). Regimes are memoryless, so a 50-session regime lasts 100+ sessions only ~13% of the time.">
              <span className="underline decoration-dotted underline-offset-4">Market regime</span>
            </Tooltip>
          }
          description={`${REGIME[r.current ?? ""] ?? "—"} for ${r.age_sessions ?? "—"} sessions. Expected durations: calm ${num(dur.calm, 0)}, normal ${num(dur.normal, 0)}, stress ${num(dur.stress, 0)} sessions.`}
          header={
            <HoverEffect className="grid grid-cols-2 gap-1">
              <Stat label="P(stress ≤ 5 sessions)" value={pct(r.p_stress_within_5, 1)} />
              <Stat label="P(stress ≤ 10 sessions)" value={pct(r.p_stress_within_10, 1)} tone={fin(r.p_stress_within_10) && r.p_stress_within_10! > 0.2 ? "bad" : undefined} />
              <Stat label="P(still in regime, 20s)" value={pct(r.p_still_in_regime?.["20"], 0)} />
              <Stat label="Worst day, last 10" value={signedPct(shock?.worst_recent_day, 1)} tone={shock?.active ? "bad" : undefined} />
            </HoverEffect>
          }
        />
        <BentoGridItem
          className="md:col-span-2 dark:bg-[#1c1c1a]"
          title={
            <Tooltip content="5,000 one-year paths of the strategy's own daily returns: a block bootstrap of real days (keeps volatility clustering) and Student-t shocks (fat tails). The key number is the chance of breaching the maximum-loss limit.">
              <span className="underline decoration-dotted underline-offset-4">Monte Carlo of the strategy</span>
            </Tooltip>
          }
          description={
            mc?.status === "ok"
              ? `Limit: a ${pct(mc.max_loss_limit, 0)} drawdown. Ruin = losing ${pct(mc.ruin_level, 0)}: bootstrap ${pct(mc.bootstrap?.p_ruin, 2)}, analytic ${pct(mc.analytic_ruin, 2)}. Longest losing streak in a year: median ${num(mc.losing_streak_1y?.p50, 0)}, 95% ${num(mc.losing_streak_1y?.p95, 0)}.`
              : "Not enough history yet."
          }
          header={
            <div className="space-y-3">
              <McBlock title="Block bootstrap (real days)" m={mc?.bootstrap} />
              <McBlock title="Student-t (fat tails)" m={mc?.student_t} />
            </div>
          }
        />
        <BentoGridItem
          className="md:col-span-2 dark:bg-[#1c1c1a]"
          title={
            <Tooltip content="Expected Shortfall of today's paper book with the correlations of market-stress days: results tested one by one say nothing about the day everything falls together.">
              <span className="underline decoration-dotted underline-offset-4">Live paper book</span>
            </Tooltip>
          }
          description={
            book && book.positions
              ? `${book.positions} positions. Worst historical day for today's weights: ${signedPct(book.worst_day_book?.return, 2)} on ${longDate(book.worst_day_book?.date)}.`
              : "No open positions."
          }
          header={
            book && book.positions ? (
              <div className="space-y-3">
                <HoverEffect className="grid grid-cols-2 gap-1 md:grid-cols-4">
                  <Stat label="Gross" value={pct(book.gross, 1)} />
                  <Stat label="Net" value={signedPct(book.net, 1)} />
                  <Stat label="ES 99% (stress corr.)" value={pct(book.es99_1d_stressed_corr, 2)} />
                  <Stat label="ES 99% (all fall)" value={pct(book.es99_1d_all_fall_together, 2)} />
                </HoverEffect>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(book.by_bucket).map(([b, v]) => (
                    <Pill key={b}>{BUCKET[b] ?? SECTOR_EN[b] ?? b}: {pct(v, 1)}</Pill>
                  ))}
                </div>
              </div>
            ) : (
              <span />
            )
          }
        />
        <BentoGridItem
          className="md:col-span-1 dark:bg-[#1c1c1a]"
          title={
            <Tooltip content="GARCH(1,1) Student-t on SPY. Persistence near 1 means shocks fade slowly; when it is ~1 the long-run variance is not identified and the sample volatility is used instead.">
              <span className="underline decoration-dotted underline-offset-4">Market volatility</span>
            </Tooltip>
          }
          description={`Half-life of a volatility shock: ${num(mkt.shock_half_life_sessions, 0)} sessions. Long-run reference: ${mkt.long_run_source ?? "—"}.`}
          header={
            <HoverEffect className="grid grid-cols-2 gap-1">
              <Stat label="Vol now (ann.)" value={pct(mkt.vol_now_annual, 1)} tone={mkt.alert ? "bad" : undefined} />
              <Stat label="Long run" value={pct(mkt.vol_long_run_annual, 1)} sub={`critical ${pct(mkt.critical_level_annual, 0)}`} />
              <Stat label="Persistence" value={num(mkt.persistence, 3)} />
              <Stat label="Tail ν" value={num(mkt.nu, 1)} sub="5–7 is fat-tailed" />
            </HoverEffect>
          }
        />
      </BentoGrid>
      {risk.review_horizon?.p_positive_after && (
        <p className="mt-4 text-xs text-neutral-500">
          Review horizon: with the backtest&apos;s edge, P(cumulative result &gt; 0) after 30 / 100 / 250 trades is{" "}
          {pct(risk.review_horizon.p_positive_after["30"], 0)} / {pct(risk.review_horizon.p_positive_after["100"], 0)} /{" "}
          {pct(risk.review_horizon.p_positive_after["250"], 0)}; {num(risk.review_horizon.trades_for_95pct, 0)} trades for 95%. The edge grows with the
          number of trades, the noise only with its square root: do not re-tune on a few red days.
        </p>
      )}
    </Panel>
  );
}

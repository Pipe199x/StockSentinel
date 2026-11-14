"use client";

import { Tabs } from "@/components/ui/tabs";
import { LineChart } from "@/components/charts/line-chart";
import { num, pct, signedPct, REGIME, SIDE, STRATEGY, SECTOR_EN, longDate, fin } from "@/lib/format";
import type { PricePoint, Signal } from "@/lib/data";
import { HoverEffect } from "@/components/ui/card-hover-effect";
import { Panel, Pill, Stat } from "./panel";

function why(s: Signal) {
  if (s.market_regime === "stress") return "The market (SPY) is in a stress regime, so the system stays flat by design.";
  if (s.strategy === "none") return "No setup: neither the momentum nor the mean-reversion rule fires today.";
  if (!s.kelly_size) return `The ${STRATEGY[s.strategy].toLowerCase()} rule fired, but the meta-model's P(success) of ${pct(s.p_success, 0)} does not beat Kelly, so the size is zero.`;
  return `${STRATEGY[s.strategy]} setup: the meta-model gives ${pct(s.p_success, 0)} to reach take-profit before the stop, after costs.`;
}

export function AssetDetail({ signal, series }: { signal: Signal | null; series: PricePoint[] }) {
  if (!signal) return null;
  const s = signal;
  const pts = series.map((p) => ({ date: p.date, value: p.close }));
  const first = pts[0]?.value;
  const last = pts[pts.length - 1]?.value;
  const change = fin(first) && fin(last) ? last / first - 1 : null;
  const on = s.kelly_size > 0;

  const signalTab = (
    <div className="h-full space-y-4 overflow-y-auto rounded-2xl border border-neutral-200 bg-white p-4 dark:border-white/[0.08] dark:bg-[#101124]">
      <p className="text-sm text-neutral-600 dark:text-neutral-300">{why(s)}</p>
      <HoverEffect className="grid grid-cols-2 gap-1">
        <Stat label="Side" value={SIDE[String(s.side)]} tone={on ? (s.side === 1 ? "good" : "bad") : undefined} />
        <Stat label="P(success)" value={pct(s.p_success, 0)} />
        <Stat label="Size" value={on ? pct(s.kelly_size) : "—"} sub={on ? (s.size_limit === "es" ? "capped by ES" : "½ Kelly × vol target") : undefined} />
        <Stat label="Exit by" value={s.t_max_date ? longDate(s.t_max_date) : "—"} sub={s.t_max_days ? `${s.t_max_days} sessions` : undefined} />
        <Stat label="Hurst" value={num(s.hurst, 2)} sub={fin(s.hurst) ? (s.hurst! >= 0.5 ? "persistent (trend)" : "anti-persistent (revert)") : undefined} />
        <Stat label="OU z-score" value={num(s.zscore, 2)} />
        <Stat label="Half-life" value={fin(s.half_life) ? `${num(s.half_life, 1)} d` : "—"} />
        <Stat label="Asset regime" value={REGIME[s.regime ?? ""] ?? "—"} sub={`P(stress) ${pct(s.p_stress, 0)}`} />
      </HoverEffect>
    </div>
  );

  const riskTab = (
    <div className="h-full space-y-4 overflow-y-auto rounded-2xl border border-neutral-200 bg-white p-4 dark:border-white/[0.08] dark:bg-[#101124]">
      <p className="text-sm text-neutral-600 dark:text-neutral-300">Fat tails only: Student-t GARCH, Cornish-Fisher and a t-distributed Monte Carlo. Never a normal distribution.</p>
      <HoverEffect className="grid grid-cols-2 gap-1">
        <Stat label="GARCH σ (daily)" value={pct(s.sigma_garch, 2)} sub={`persistence ${num(s.garch_persistence, 3)}`} />
        <Stat label="Tail ν" value={num(s.garch_nu, 1)} sub="lower = fatter tails" />
        <Stat label="VaR 95% (1d)" value={pct(s.var_95, 2)} />
        <Stat label="ES 99% (1d)" value={pct(s.es_99, 2)} />
        <Stat label="Cornish-Fisher VaR" value={pct(s.var_95_cornish_fisher, 2)} />
        <Stat label={`MC ${s.mc_horizon_days ?? 20}d median`} value={signedPct(s.mc_p50)} />
        <Stat label="MC 5–95%" value={`${signedPct(s.mc_p05, 0)} / ${signedPct(s.mc_p95, 0)}`} />
        <Stat label="P(loss > 10%)" value={pct(s.mc_prob_loss_10pct, 1)} />
      </HoverEffect>
    </div>
  );

  const questionsTab = (
    <div className="h-full space-y-3 overflow-y-auto rounded-2xl border border-neutral-200 bg-white p-4 text-sm dark:border-white/[0.08] dark:bg-[#101124]">
      <p className="text-xs text-neutral-500">The four questions to answer before risking a dollar.</p>
      <div>
        <p className="font-semibold">1. How is the profit generated?</p>
        <p className="text-neutral-600 dark:text-neutral-300">{s.thesis ?? why(s)}</p>
      </div>
      <div>
        <p className="font-semibold">2. What do I lose in the worst case?</p>
        <p className="text-neutral-600 dark:text-neutral-300">
          {on
            ? `If the stop is hit: about ${pct(s.risk_to_stop_pct_equity, 2)} of equity. The average of the worst 1% outcomes over the trade (Expected Shortfall): ${pct(s.es_trade_pct_equity, 2)} of equity, capped at 1% by design.`
            : "Nothing today: there is no position."}
          {fin(s.market_p_stress_10) && ` Chance the market enters a stress regime within 10 sessions: ${pct(s.market_p_stress_10, 0)}.`}
        </p>
      </div>
      <div>
        <p className="font-semibold">3. Do I understand the system behind it?</p>
        <p className="text-neutral-600 dark:text-neutral-300">
          A fixed, pre-declared rule plus a meta-model, validated out of sample (see Validation). Every change is logged as a trial.
        </p>
      </div>
      <div>
        <p className="font-semibold">4. Who else gets hurt if it goes wrong?</p>
        <p className="text-neutral-600 dark:text-neutral-300">Nobody: this is simulated paper trading, no real money.</p>
      </div>
    </div>
  );

  return (
    <Panel
      id="asset"
      eyebrow={SECTOR_EN[s.sector ?? ""] ?? s.sector ?? "Asset"}
      title={s.ticker}
      aside={
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-2xl font-semibold tabular-nums">{num(s.last_close)}</span>
          <span className="text-xs text-neutral-500">USD</span>
          {change != null && <Pill tone={change >= 0 ? "good" : "bad"}>{signedPct(change)} · 60 sessions</Pill>}
        </div>
      }
    >
      <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr]">
      <LineChart
        label={`${s.ticker} closing prices over the last 60 sessions with the current take-profit and stop-loss`}
        points={pts}
        levels={[
          ...(s.tp ? [{ value: s.tp, label: "TP", tone: "gain" as const }] : []),
          ...(s.sl ? [{ value: s.sl, label: "SL", tone: "loss" as const }] : []),
        ]}
      />
      <div className="h-[34rem] sm:h-[25rem]">
        <Tabs
          tabs={[
            { title: "Signal", value: "signal", content: signalTab },
            { title: "Risk", value: "risk", content: riskTab },
            { title: "4 questions", value: "questions", content: questionsTab },
          ]}
          activeTabClassName="bg-cyan-500/15 dark:bg-cyan-400/15"
          contentClassName="mt-4"
        />
      </div>
      </div>
    </Panel>
  );
}

"use client";

import { useCallback, useEffect, useState } from "react";

/* Shapes of the JSON files the Python pipeline publishes to docs/data/. */

export type Regime = "calm" | "normal" | "stress" | null;

export type Signal = {
  ticker: string;
  as_of: string;
  last_close: number;
  regime: Regime;
  p_stress: number | null;
  market_regime: Regime;
  market_p_stress: number | null;
  hurst: number | null;
  half_life: number | null;
  zscore: number | null;
  strategy: "none" | "momentum" | "mean_reversion";
  side: -1 | 0 | 1;
  p_success: number | null;
  kelly_size: number;
  size_limit?: "kelly" | "es" | null;
  tp: number | null;
  sl: number | null;
  t_max_days: number | null;
  t_max_date: string | null;
  sigma_garch?: number | null;
  garch_persistence?: number | null;
  garch_nu?: number | null;
  var_95?: number | null;
  es_99?: number | null;
  var_95_cornish_fisher?: number | null;
  mc_horizon_days?: number | null;
  mc_p05?: number | null;
  mc_p50?: number | null;
  mc_p95?: number | null;
  mc_prob_loss_10pct?: number | null;
  sector?: string | null;
  sigma_next?: number | null;
  data_stale?: boolean;
  /* the advisor's four questions before investing */
  thesis?: string | null;
  risk_to_stop_pct_equity?: number | null;
  es_trade_pct_equity?: number | null;
  market_p_stress_10?: number | null;
};

export type TradeStats = {
  n_trades: number;
  years?: number;
  win_rate?: number;
  avg_win?: number;
  avg_loss?: number;
  expected_value?: number;
  sr_per_trade?: number;
  sr_annualized?: number;
  skew?: number;
  kurtosis?: number;
  max_losing_streak?: number;
  dsr?: number | null;
  expected_streak?: number;
  streak_drawdown_1pct?: number;
  streak_drawdown_5pct?: number;
  min_trades_for_edge?: number;
  calendar?: CalendarStats;
  equity_curve_weekly?: { date: string; equity: number }[];
  effective_n?: number;
  mean_uniqueness?: number | null;
  dsr_effective?: number | null;
  sr_ci?: { sr: number | null; se: number | null; ci: [number, number] | null };
  min_track_record_trades?: number | null;
  streak_threshold_30?: number | null;
};

export type CalendarStats = {
  sessions?: number;
  years?: number;
  total_return?: number;
  cagr?: number | null;
  volatility?: number;
  sharpe?: number | null;
  sharpe_excess_rf?: number | null;
  risk_free_used?: boolean;
  sortino?: number | null;
  sortino_to_sharpe?: number | null;
  max_drawdown?: number;
  calmar?: number | null;
  time_in_market?: number;
  avg_gross_exposure?: number;
  profit_factor?: number | null;
  payoff_ratio?: number | null;
};

export type RedFlag = { code: string; severity: "info" | "warning" | "critical"; message: string; value?: unknown; threshold?: unknown };
export type BucketStats = { n: number; win_rate?: number; expected_value?: number; sr_per_trade?: number | null; pnl_sum?: number };

export type StressWindow = {
  from: string;
  to: string;
  covered: boolean;
  benchmark_return: number | null;
  n_trades: number;
  win_rate: number | null;
  expected_value: number | null;
  pnl_sum: number;
};

export type ValidationReport = {
  as_of: string;
  generated_utc: string;
  folds: { test_from: string; test_to: string; n_train: number; n_test: number; accuracy: number; sr_per_trade: number }[];
  cpcv_path_sr: number[];
  meta: TradeStats;
  primary: TradeStats;
  n_events: number;
  n_trials: number;
  dsr_threshold: number;
  edge_is_significant: boolean;
  trades_by_regime: Record<string, number>;
  feature_importance: Record<string, number>;
  correlation: { all_days: number | null; market_stress_days: number | null; market_calm_days: number | null; n_stress_days: number };
  stress_windows?: Record<string, StressWindow>;
  config: Record<string, unknown>;
  limitations: string[];
  config_hash?: string;
  strategy_label?: string | null;
  red_flags?: RedFlag[];
  by_regime?: Record<string, BucketStats>;
  scenario_coverage?: { scenarios: Record<string, BucketStats>; undercovered: string[] };
  stability?: { runs: { seed: number; n_trades: number; sr_per_trade: number; expected_value: number; dsr: number | null }[]; sr_range: [number, number] | null; dsr_range: [number, number] | null };
  ffd?: { tickers: Record<string, { d: number; memory_corr: number; adf_p_traded: number | null }>; non_stationary_traded: string[] };
  data_freshness?: { prices_as_of: string; news_fetched_utc: string | null; news_fresh: boolean };
  environment?: Record<string, string>;
};

export type GarchDiag = {
  persistence?: number; stationary?: boolean; nu?: number | null; vol_now_annual?: number | null; vol_long_run_annual?: number | null;
  long_run_source?: string | null; ratio_to_long_run?: number | null; critical_level_annual?: number | null; alert?: boolean; shock_half_life_sessions?: number | null;
};
export type McStats = {
  p_profit: number; p_breach_max_loss: number; p_ruin: number; drawdown_p50: number; drawdown_p95: number; drawdown_p99: number;
  return_p05: number; return_p50: number; return_p95: number; var95: number; cvar95: number;
};
export type RiskReport = {
  as_of: string;
  config_hash?: string;
  regime?: {
    current?: string; age_sessions?: number; expected_duration_sessions?: Record<string, number | null>;
    p_stay_next_session?: number; p_still_in_regime?: Record<string, number>; p_stress_within_5?: number; p_stress_within_10?: number;
  };
  training_mix?: { share?: Record<string, number>; current_regime_rare_in_training?: boolean };
  garch?: { market?: GarchDiag; tickers?: Record<string, GarchDiag> };
  garch_alerts?: string[];
  post_shock?: { active: boolean; worst_recent_day?: number | null; sessions_since_shock?: number; window_sessions?: number };
  montecarlo?: {
    status: string; paths?: number; horizon_days?: number; max_loss_limit?: number; ruin_level?: number;
    bootstrap?: McStats; student_t?: McStats; losing_streak_1y?: { trades_per_year?: number; p50?: number; p95?: number }; analytic_ruin?: number;
  };
  book?: {
    positions: number; gross: number; net: number; by_bucket: Record<string, number>; es99_1d_stressed_corr?: number;
    es99_1d_all_fall_together?: number; worst_day_book?: { date: string; return: number } | null;
  };
  review_horizon?: { p_positive_after?: Record<string, number>; trades_for_95pct?: number | null };
};

export type PricePoint = { date: string; close: number; adj_close?: number; volume?: number };
export type Prices = { as_of: string; window_days: number; series: Record<string, PricePoint[]> };

export type Article = {
  published_at: string;
  source: string;
  title: string;
  summary?: string;
  url: string;
  image_url?: string | null;
  tickers: string[];
  sentiment?: { label: "positive" | "negative" | "neutral"; score: number };
};
export type News = { as_of: string; window_days: number; articles: Article[] };

export type DeepModel = TradeStats & { promoted?: boolean; n_trials?: number };
export type DeepReport = {
  as_of: string;
  xgb?: DeepModel;
  lstm?: DeepModel;
  fusion?: { status?: string; promoted?: boolean; events_with_news_coverage?: number; min_events?: number; news_tickers?: string[] } & Partial<DeepModel>;
};

export type PaperRow = {
  ticker: string;
  signal_date: string;
  side: number;
  size: number;
  p_success: number | null;
  tp: number;
  sl: number;
  exit_by: string;
  status: "pending" | "open" | "closed" | "missed";
  entry_date: string | null;
  entry_price: number | null;
  exit_date: string | null;
  exit_price: number | null;
  exit_reason: "tp" | "sl" | "time" | "entry_gap" | "late_publish" | null;
  ret_net: number | null;
  config_hash?: string | null;
  strategy_label?: string | null;
  div_cash?: number | null;
};

export type PaperSummary = {
  as_of: string;
  started: string | null;
  counts: { pending: number; open: number; closed: number; missed?: number };
  config_hash?: string | null;
  current_closed?: number;
  by_config?: { config_hash: string; strategy_label: string | null; first_signal: string; closed: number; open: number; missed: number; expected_value: number | null }[];
  gate1?: { status: string; p_breakeven: number | null; p_backtest: number | null; trades_required: number | null; live_trades: number; live_effective_trades: number; years_to_go: number | null; p_value: number | null };
  rescore?: { ticker: string; signal_date: string; sessions_elapsed: number; pnl_sigma: number; p_entry: number | null; n_similar: number; p_tp_now: number | null; p_profit_now: number | null }[];
  win_rate: number | null;
  expected_value: number | null;
  sr_per_trade: number | null;
  total_return: number;
  equity_curve: { date: string; equity: number }[];
  /** Locked (public) file: open rows keep only side, status and exit_by. */
  positions: PaperRow[];
  /** Public file only: share of the account in open positions (sizes are redacted). */
  invested?: number;
  recent_closed: PaperRow[];
  gate0: {
    status: "collecting" | "pass" | "fail" | "inconclusive";
    reason: string;
    min_trades: number;
    n_closed?: number;
    sr_ci95?: [number, number] | null;
    min_detectable_gap?: number | null;
    backtest_cpcv_median?: number | null;
    live_sr_per_trade: number | null;
    backtest_sr_per_trade: number | null;
    backtest_expected_value: number | null;
    backtest_cpcv_range: [number, number] | null;
  };
};

export type Universe = { sectors: Record<string, string[]>; benchmark: string };
export type Manifest = { stocks?: { last_updated_utc: string; as_of: string }; news?: { last_updated_utc: string } };

export type Dashboard = {
  /** true when the premium files were refused: tickers, levels and odds below are placeholders. */
  locked: boolean;
  signals: Signal[];
  report: ValidationReport | null;
  prices: Prices | null;
  news: News | null;
  deep: DeepReport | null;
  paper: PaperSummary | null;
  risk: RiskReport | null;
  universe: Universe | null;
  manifest: Manifest | null;
};

/** Public signals_latest.json (stocks_etl/publish_json.redact_signals): no ticker, level, odds or size. */
type PublicSignals = {
  as_of: string;
  market_regime: Regime;
  market_p_stress: number | null;
  n_assets: number;
  n_idle: number;
  picks: Pick<Signal, "as_of" | "side" | "strategy" | "t_max_date" | "t_max_days" | "market_regime" | "market_p_stress">[];
};

/**
 * Placeholder signals for the locked page, so every component keeps one shape. The values are
 * deliberately meaningless; components check `locked` and show blurred dummy text instead.
 */
function placeholderSignals(p: PublicSignals): Signal[] {
  const blank = {
    last_close: 0, regime: null, p_stress: null, hurst: null, half_life: null, zscore: null, p_success: null,
    tp: null, sl: null, sector: null,
  };
  const picks = p.picks.map((s, i) => ({ ...blank, ...s, ticker: `PICK ${i + 1}`, kelly_size: 1 }) as Signal);
  const idle = Array.from({ length: p.n_idle }, (_, i) =>
    ({ ...blank, ticker: `ASSET ${i + 1}`, as_of: p.as_of, side: 0, strategy: "none", kelly_size: 0, t_max_days: null, t_max_date: null,
      market_regime: p.market_regime, market_p_stress: p.market_p_stress }) as Signal,
  );
  return [...picks, ...idle];
}

const BASE = "/data/";

async function getJSON<T>(name: string): Promise<T | null> {
  try {
    const r = await fetch(`${BASE}${name}?t=${Date.now()}`, { cache: "no-store", credentials: "same-origin" });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

export function useDashboard() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    // Premium first: the Pages Function answers 401 without a valid session (functions/data/_middleware.ts).
    const [fullSignals, fullPaper, deep, risk, report, prices, news, universe, manifest] = await Promise.all([
      getJSON<Signal[]>("premium/signals_latest.json"),
      getJSON<PaperSummary>("premium/paper_summary.json"),
      getJSON<DeepReport>("premium/deep_report.json"),
      getJSON<RiskReport>("premium/risk_report.json"),
      getJSON<ValidationReport>("validation_report.json"),
      getJSON<Prices>("prices_latest.json"),
      getJSON<News>("news_latest.json"),
      getJSON<Universe>("universe.json"),
      getJSON<Manifest>("manifest.json"),
    ]);
    const locked = !Array.isArray(fullSignals);
    let signals: Signal[] = fullSignals ?? [];
    let paper = fullPaper;
    if (locked) {
      const [pub, pubPaper] = await Promise.all([getJSON<PublicSignals>("signals_latest.json"), getJSON<PaperSummary>("paper_summary.json")]);
      signals = pub?.picks ? placeholderSignals(pub) : [];
      paper = pubPaper;
      if (!pub) setError("Could not load today's picks.");
      else setError(null);
    } else setError(null);
    setData({ locked, signals, report, prices, news, deep: locked ? null : deep, paper, risk: locked ? null : risk, universe, manifest });
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  /** Checks a Gumroad license key server-side; on success the session cookie unlocks /data/premium. */
  const unlock = useCallback(
    async (key: string): Promise<string | null> => {
      try {
        const r = await fetch("/api/unlock", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ license_key: key }),
          credentials: "same-origin",
        });
        const body = (await r.json().catch(() => ({}))) as { error?: string };
        if (!r.ok) return body.error ?? "That license key did not work.";
        await load();
        return null;
      } catch {
        return "Could not reach the server. Check your connection and try again.";
      }
    },
    [load],
  );

  const logout = useCallback(async () => {
    try {
      await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
    } catch {}
    await load();
  }, [load]);

  return { data, error, loading, reload: load, unlock, logout };
}

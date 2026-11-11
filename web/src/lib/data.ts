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
};

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
  status: "pending" | "open" | "closed";
  entry_date: string | null;
  entry_price: number | null;
  exit_date: string | null;
  exit_price: number | null;
  exit_reason: "tp" | "sl" | "time" | "entry_gap" | null;
  ret_net: number | null;
};

export type PaperSummary = {
  as_of: string;
  started: string | null;
  counts: { pending: number; open: number; closed: number };
  win_rate: number | null;
  expected_value: number | null;
  sr_per_trade: number | null;
  total_return: number;
  equity_curve: { date: string; equity: number }[];
  positions: PaperRow[];
  recent_closed: PaperRow[];
  gate0: {
    status: "collecting" | "pass" | "fail";
    reason: string;
    min_trades: number;
    live_sr_per_trade: number | null;
    backtest_sr_per_trade: number | null;
    backtest_expected_value: number | null;
    backtest_cpcv_range: [number, number] | null;
  };
};

export type Universe = { sectors: Record<string, string[]>; benchmark: string };
export type Manifest = { stocks?: { last_updated_utc: string; as_of: string }; news?: { last_updated_utc: string } };

export type Dashboard = {
  signals: Signal[];
  report: ValidationReport | null;
  prices: Prices | null;
  news: News | null;
  deep: DeepReport | null;
  paper: PaperSummary | null;
  universe: Universe | null;
  manifest: Manifest | null;
};

const BASE = "/data/";

async function getJSON<T>(name: string): Promise<T | null> {
  try {
    const r = await fetch(`${BASE}${name}?t=${Date.now()}`, { cache: "no-store" });
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
    const [signals, report, prices, news, deep, paper, universe, manifest] = await Promise.all([
      getJSON<Signal[]>("signals_latest.json"),
      getJSON<ValidationReport>("validation_report.json"),
      getJSON<Prices>("prices_latest.json"),
      getJSON<News>("news_latest.json"),
      getJSON<DeepReport>("deep_report.json"),
      getJSON<PaperSummary>("paper_summary.json"),
      getJSON<Universe>("universe.json"),
      getJSON<Manifest>("manifest.json"),
    ]);
    if (!signals) setError("Could not load the published signals.");
    else setError(null);
    setData({ signals: signals ?? [], report, prices, news, deep, paper, universe, manifest });
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { data, error, loading, reload: load };
}

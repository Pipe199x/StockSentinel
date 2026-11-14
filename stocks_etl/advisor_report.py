"""The advisor's measurement layer: turns the out-of-sample backtest, the regime models and the
live paper book into the numbers his videos ask for (research/advisor/COMPLIANCE.md maps each
one). Informational only: it never changes a signal, a size or a trade.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from etl_common.risk_free import load_daily_rf
from stocks_etl.book import book_risk, edge_decay, progress_table, rescore_positions, review_horizon
from stocks_etl.meta_model import STRESS_WINDOWS, stability
from stocks_etl.metrics import (average_uniqueness, calendar_returns, calendar_stats, daily_pnl, equity_weekly,
                                gate1, min_track_record, red_flags, sr_ci, streak_threshold)
from stocks_etl.montecarlo import strategy_mc
from stocks_etl.regime_analytics import REGIMES, garch_diag, post_shock, training_mix, transition_summary
from stocks_etl.validation import deflated_sharpe

DEFAULT_REPORT = {"max_loss_limit": 0.10, "ruin_level": 0.30, "strategy_mc_paths": 5000, "mc_block": 20,
                  "mc_horizon_days": 252, "stability_seeds": [7, 11, 13],
                  "gates": {"gate1_alpha": 0.05, "gate1_power": 0.80}, "flags": {}}


def _bucket_stats(t: pd.DataFrame) -> Dict:
    if t.empty:
        return {"n": 0}
    r = t["ret_net"].astype(float)
    pnl = t["pnl"].astype(float)
    sd = pnl.std(ddof=1)
    return {"n": int(len(t)), "win_rate": float((r > 0).mean()), "expected_value": float(r.mean()),
            "sr_per_trade": float(pnl.mean() / sd) if len(t) > 1 and sd > 0 else None, "pnl_sum": float(pnl.sum())}


def by_regime(trades: pd.DataFrame) -> Dict:
    if trades.empty:
        return {}
    reg = trades["regime"].dropna().astype(int).map(REGIMES)
    return {k: _bucket_stats(trades.loc[reg.index[reg == k]]) for k in sorted(reg.unique())}


def scenario_coverage(trades: pd.DataFrame, feats: pd.DataFrame, bench: str) -> Dict:
    """Scenarios the backtest has lived through, not just its trade count (20 Oct video)."""
    spy = feats.loc[feats["ticker"] == bench, ["date", "close", "rv_20"]].copy()
    if trades.empty or spy.empty:
        return {}
    spy["day"] = pd.to_datetime(spy["date"]).dt.tz_localize(None).dt.normalize()
    spy = spy.set_index("day").sort_index()
    trend = spy["close"].pct_change(120)
    hi_vol = spy["rv_20"] >= spy["rv_20"].quantile(0.9)
    start = pd.DatetimeIndex(pd.to_datetime(trades["start"])).normalize()
    tr = trend.reindex(start).to_numpy()
    hv = hi_vol.reindex(start).fillna(False).to_numpy()
    cases = {"bull (SPY 120d > +5%)": tr > 0.05, "bear (SPY 120d < -5%)": tr < -0.05,
             "sideways": np.abs(tr) <= 0.05, "volatility spike (SPY rv20 top decile)": hv}
    for name, (a, b) in STRESS_WINDOWS.items():
        cases[name] = (start >= pd.Timestamp(a)) & (start <= pd.Timestamp(b))
    out = {k: _bucket_stats(trades[np.asarray(m, bool)]) for k, m in cases.items()}
    return {"scenarios": out, "undercovered": [k for k, v in out.items() if v.get("n", 0) < 30]}


def ffd_report(feats: pd.DataFrame, fmeta: Dict) -> Dict:
    """Memory kept by fractional differencing (corr with log price) and stationarity on the
    traded period (ADF with AIC lags): the advisor's 'test stationarity first'."""
    from statsmodels.tsa.stattools import adfuller
    rows = {}
    for t, g in feats.groupby("ticker"):
        g = g[g["sigma_garch"].notna() & g["ffd_close"].notna()]
        if len(g) < 100:
            continue
        x = g["ffd_close"].to_numpy(float)
        try:
            p = float(adfuller(x, autolag="AIC")[1])
        except Exception:  # noqa: BLE001
            p = None
        rows[t] = {"d": fmeta.get(t, {}).get("ffd_d"),
                   "memory_corr": float(np.corrcoef(x, np.log(g["close"].to_numpy(float)))[0, 1]),
                   "adf_p_traded": p}
    bad = [t for t, v in rows.items() if v["adf_p_traded"] is not None and v["adf_p_traded"] >= 0.05]
    return {"tickers": rows, "non_stationary_traded": bad}


def build(prices: pd.DataFrame, feats: pd.DataFrame, fmeta: Dict, internals: Dict, report: Dict,
          signals: pd.DataFrame, ledger: pd.DataFrame, paper: Dict, cfg: dict, docs_dir: str) -> Dict:
    qcfg = cfg.get("quant", {}) or {}
    rcfg = {**DEFAULT_REPORT, **(cfg.get("report") or {})}
    bench = qcfg.get("benchmark", "SPY")
    ev, trades = internals["events"], internals["meta_trades"]
    m = report["meta"]
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize().unique()))
    rf = load_daily_rf(os.path.join(docs_dir, "history", "risk_free.csv"))

    # --- backtest on a calendar-time portfolio ------------------------------------------------
    cal = calendar_returns(daily_pnl(trades, prices, qcfg.get("cost_bps", 5)), sessions)
    cstats = calendar_stats(cal, trades, rf)
    pnl = trades["pnl"].to_numpy(float) if len(trades) else np.array([])
    uniq = average_uniqueness(trades["start"], trades["end"], sessions) if len(trades) else np.array([])
    n_eff = float(uniq.sum())
    dsr_eff = (deflated_sharpe(m["sr_per_trade"], report["n_trials"], report["var_sr_used"], m["skew"],
                               m["kurtosis"], max(int(round(n_eff)), 2)) if n_eff > 2 else None)
    meta_add = {"calendar": cstats, "equity_curve_weekly": equity_weekly(cal), "effective_n": n_eff,
                "mean_uniqueness": float(uniq.mean()) if len(uniq) else None, "dsr_effective": dsr_eff,
                "sr_ci": sr_ci(pnl), "min_track_record_trades": min_track_record(pnl),
                "streak_threshold_30": streak_threshold(m.get("win_rate", 0.5), 30)}
    years = cstats.get("years") or 0
    tpy = len(trades) / years if years else None

    # --- regimes and volatility ---------------------------------------------------------------
    mkt = feats[feats["ticker"] == bench].sort_values("date")
    hmm = (fmeta.get(bench) or {}).get("hmm")
    p_now = mkt[["p_calm", "p_normal", "p_stress"]].iloc[-1].to_numpy(float) if len(mkt) else None
    regime = transition_summary(hmm["transmat"], mkt["regime"], p_now) if hmm and len(mkt) else {}
    sig = signals.set_index("ticker")
    sample_sd = feats.groupby("ticker")["log_ret_1"].std()
    garch = {t: garch_diag(fm.get("garch") or {}, sig["sigma_next"].get(t), sample_sigma=sample_sd.get(t))
             for t, fm in fmeta.items()}

    # --- the live paper book (current configuration only for the statistics) ------------------
    cur = ledger[(ledger["config_hash"] == report.get("config_hash")) & (ledger["status"] == "closed")]
    live_ret = cur["ret_net"].astype(float).to_numpy()
    live_neff = (float(average_uniqueness(cur["entry_date"], cur["exit_date"], sessions).sum()) if len(cur) else 0.0)
    sectors = {t: s for s, ts in (cfg.get("universe") or {}).items() for t in ts}
    # re-scoring is conditioned on the trades the meta-model actually took (not every primary event)
    taken = trades.merge(ev[["ticker", "start", "sigma_event"]], on=["ticker", "start"], how="left") if len(trades) else trades
    table = progress_table(taken, prices)
    risk = {
        "as_of": report.get("as_of") or str(sessions[-1].date()), "config_hash": report.get("config_hash"),
        "regime": regime, "training_mix": training_mix(trades, regime.get("current")),
        "garch": {"market": garch.get(bench, {}), "tickers": garch},
        "garch_alerts": sorted(t for t, g in garch.items() if g.get("alert")),
        "post_shock": post_shock(mkt.set_index("date")["log_ret_1"]),
        "montecarlo": strategy_mc(cal["pnl"].to_numpy(float) if len(cal) else np.array([]), trades, rcfg,
                                  int(qcfg.get("seed", 42))),
        "book": book_risk(ledger, signals, feats, sectors),
        "review_horizon": review_horizon(pnl),
    }
    trade_sd = float(trades["ret_net"].std(ddof=1)) if len(trades) > 1 else None
    g = rcfg.get("gates", {})
    paper_add = {
        "gate1": gate1(m, live_ret, live_neff, tpy, g.get("gate1_alpha", 0.05), g.get("gate1_power", 0.80)),
        "edge_decay": edge_decay(pd.Series(live_ret), m.get("expected_value"), trade_sd),
        "live_sr_ci": sr_ci((cur["size"].astype(float) * cur["ret_net"].astype(float)).to_numpy()) if len(cur) else None,
        "rescore": rescore_positions(ledger, prices, table, float(qcfg.get("k_tp", 1.5)), float(qcfg.get("k_sl", 1.5))),
    }
    validation_add = {
        "by_regime": by_regime(trades), "scenario_coverage": scenario_coverage(trades, feats, bench),
        "stability": stability(ev, qcfg, rcfg.get("stability_seeds", [7, 11, 13]), report["n_trials"],
                               report["var_sr_used"]),
        "ffd": ffd_report(feats, fmeta),
    }
    validation_add["red_flags"] = red_flags(m, cstats, report, {t: {"persistence": v.get("persistence"), "nu": v.get("nu")}
                                                               for t, v in garch.items()}, rcfg)
    if not len(rf):
        validation_add["red_flags"].append({"code": "risk_free_missing", "severity": "info",
                                            "message": "No risk-free series yet: Sharpe in excess of risk-free uses 0",
                                            "value": None, "threshold": None})
    return {"meta": meta_add, "validation": validation_add, "risk": risk, "paper": paper_add}

"""Replay the live paper-trading bot over the last N months: a Gate 0 verdict without waiting.

    python -m research.replay.paper_replay [--months 24] [--embargo 5] [--refresh]

What it simulates, day by day, using only what was knowable on that day:
  - features: point-in-time (yearly GARCH/HMM refits on past data; see tests/test_pit.py),
    so computing them once on the full history leaks nothing;
  - the meta-model: retrained at the start of every month on events whose triple-barrier
    label had already resolved (end < month start - embargo);
  - signals: the same sizing, take-profit, stop and exit date the live pipeline publishes;
  - execution: stocks_etl.paper.update_ledger, fed prices truncated to each day, so fills
    (next open, gaps, double touch = stop, time exit) follow the live rules exactly.

It also runs the backtest execution (sequential_trades: entry at the signal close, barriers
on closes) on the SAME probabilities, so any gap between the two is execution, not model.

Research only: nothing is written to docs/ and nothing is logged to the live trials file.
Outputs go to research/replay/out/ (gitignored).
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from stocks_etl.meta_model import (_model, feature_cols, _nth_trading_day, build_events, make_signals,
                                   position_size, sequential_trades)
from stocks_etl.paper import COLUMNS, summarize, update_ledger
from stocks_etl.primary import primary_side
from stocks_etl.validation import sharpe

OUT = Path("research/replay/out")
YF_START = "2005-01-01"  # the replay results on record were computed on this history


# ---------------------------------------------------------------- data

def load_prices(cfg: dict, refresh: bool) -> pd.DataFrame:
    """Same download and normalization as the daily pipeline, cached locally."""
    path = OUT / "prices.csv.gz"
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df
    from research.yf_prices import _download_with_retry, _normalize
    from stocks_etl.stocks_etl import universe_symbols
    symbols = universe_symbols(cfg)
    # research baseline: yfinance from 2005, split-adjusted only (the live pipeline is Massive-only from 2016)
    raw = _download_with_retry(symbols=symbols, start=YF_START, end=cfg.get("end"), interval="1d",
                               auto_adjust=False, actions=True, threads=1)
    df = _normalize(df_raw=raw, symbols=symbols, auto_adjust=False, cal_name=cfg.get("calendar"))
    if df.empty:
        raise SystemExit("yfinance returned no data")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_features(df: pd.DataFrame, qcfg: dict, refresh: bool):
    from stocks_etl.features import make_features
    path = OUT / "features.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    print(f"[features] {df['ticker'].nunique()} tickers (GARCH + HMM refits, the slow part)")
    fm = make_features(df, qcfg)
    pd.to_pickle(fm, path)
    return fm


# ---------------------------------------------------------------- replay pieces

def candidates(feats: pd.DataFrame, cfg: dict, start: pd.Timestamp) -> pd.DataFrame:
    """Every day from `start` on where the primary rule fires, open barrier or not."""
    out = []
    for _, f in feats.groupby("ticker", sort=False):
        f = f.sort_values("date").reset_index(drop=True)
        f = pd.concat([f, primary_side(f, cfg)], axis=1)
        f["sigma_event"] = f["sigma_garch"] * np.sqrt(f["t_max"])
        f["day"] = pd.to_datetime(f["date"]).dt.tz_localize(None).dt.normalize()
        keep = (f["side"] != 0) & (f["day"] >= start) & np.isfinite(f["sigma_event"])
        if "eligible" in f.columns:  # point-in-time universe: never trade a stock outside it
            keep &= f["eligible"].fillna(False).astype(bool)
        out.append(f[keep])
    return pd.concat(out, ignore_index=True).sort_values(["day", "ticker"]).reset_index(drop=True)


def walk_forward_probs(ev: pd.DataFrame, cand: pd.DataFrame, cfg: dict, embargo_days: int) -> np.ndarray:
    """Monthly retrain on resolved labels only. Returns P(success) for every candidate."""
    seed = int(cfg.get("seed", 42))
    prob = np.full(len(cand), np.nan)
    months = cand["day"].dt.to_period("M")
    for m in months.unique():
        m_start = m.start_time
        cutoff = m_start - pd.offsets.BDay(embargo_days)
        train = ev[ev["end"] < cutoff]
        if len(train) < 200 or train["label"].nunique() < 2:
            continue
        model = _model(seed).fit(train[feature_cols(cfg)], train["label"])
        idx = np.flatnonzero((months == m).to_numpy())
        prob[idx] = model.predict_proba(cand.iloc[idx][feature_cols(cfg)])[:, 1]
    return prob


def build_signals(cand: pd.DataFrame, prob: np.ndarray, cfg: dict, cal: str) -> pd.DataFrame:
    """Signal rows in the shape make_signals publishes (meta_model.py: tp/sl/t_max_date)."""
    k_tp, k_sl = float(cfg.get("k_tp", 1.5)), float(cfg.get("k_sl", 1.5))

    @lru_cache(maxsize=None)
    def exit_date(day: pd.Timestamp, n: int) -> str:
        return _nth_trading_day(day, n, cal)

    rows = []
    for r, p in zip(cand.itertuples(), prob):
        if not np.isfinite(p):
            continue
        nu = r.garch_nu if np.isfinite(r.garch_nu) else 5.0
        size, limit = position_size(p, r.sigma_garch, nu, r.t_max, cfg)
        if size <= 0:
            continue
        close = float(r.close)
        rows.append({"ticker": r.ticker, "as_of": str(r.day.date()), "side": int(r.side), "p_success": float(p),
                     "kelly_size": round(size, 4), "size_limit": limit, "strategy": int(r.strategy),
                     "mkt_regime": r.mkt_regime,
                     "tp": close * (1 + r.side * k_tp * r.sigma_event),
                     "sl": close * (1 - r.side * k_sl * r.sigma_event),
                     "t_max_date": exit_date(r.day, int(r.t_max))})
    return pd.DataFrame(rows)


def replay_ledger(prices: pd.DataFrame, signals: pd.DataFrame, start: pd.Timestamp, cost_bps: float) -> pd.DataFrame:
    """Run the live ledger once per session, each time seeing only prices up to that day."""
    p = prices.copy()
    p["day"] = pd.to_datetime(p["date"]).dt.tz_localize(None).dt.normalize()
    p = p[p["day"] >= start - pd.Timedelta(days=7)]
    by_day = {d: g for d, g in signals.groupby("as_of")} if len(signals) else {}
    ledger = pd.DataFrame(columns=COLUMNS)
    days = sorted(p.loc[p["day"] >= start, "day"].unique())
    for i, d in enumerate(days):
        sig = by_day.get(str(pd.Timestamp(d).date()), signals.iloc[0:0])
        ledger = update_ledger(ledger, p[p["day"] <= d].drop(columns="day"), sig, cost_bps)
        if i % 60 == 0:
            print(f"[replay] {pd.Timestamp(d).date()}  rows={len(ledger)}")
    return ledger


# ---------------------------------------------------------------- report

def _stats(ret: pd.Series, pnl: pd.Series) -> dict:
    if len(ret) == 0:
        return {"n_trades": 0}
    return {"n_trades": int(len(ret)), "win_rate": float((ret > 0).mean()), "expected_value": float(ret.mean()),
            "sr_per_trade": sharpe(pnl) if len(pnl) > 1 else None, "pnl_sum": float(pnl.sum())}


def compare(ledger: pd.DataFrame, signals: pd.DataFrame, ev: pd.DataFrame, cand: pd.DataFrame,
            prob: np.ndarray, cfg: dict) -> dict:
    closed = ledger[ledger["status"] == "closed"].copy()
    closed["ret_net"] = closed["ret_net"].astype(float)
    closed["pnl"] = closed["size"].astype(float) * closed["ret_net"]
    meta = signals[["ticker", "as_of", "strategy", "mkt_regime"]].rename(columns={"as_of": "signal_date"})
    closed = closed.merge(meta, on=["ticker", "signal_date"], how="left")
    closed["month"] = closed["exit_date"].str[:7]

    # backtest execution on the same probabilities: entry at the signal close, close barriers
    key = cand[["ticker", "day"]].assign(prob=prob)
    e = ev.assign(day=pd.to_datetime(ev["start"]).dt.normalize()).merge(key, on=["ticker", "day"], how="inner")
    bt = sequential_trades(e.drop(columns="prob"), e["prob"].to_numpy(), cfg) if len(e) else pd.DataFrame()

    names = {0: "calm", 1: "normal", 2: "stress", 1.0: "normal", 0.0: "calm", 2.0: "stress"}
    return {
        "paper_execution": _stats(closed["ret_net"], closed["pnl"]),
        "backtest_execution_same_probs": _stats(bt["ret_net"], bt["pnl"]) if len(bt) else {"n_trades": 0},
        "exit_reasons": closed["exit_reason"].value_counts().to_dict(),
        "by_strategy": {("momentum" if k == 1 else "mean_reversion"): _stats(g["ret_net"], g["pnl"])
                        for k, g in closed.groupby("strategy")},
        "by_market_regime": {names.get(k, str(k)): _stats(g["ret_net"], g["pnl"])
                             for k, g in closed.groupby("mkt_regime")},
        "by_month": {k: round(float(g["pnl"].sum()), 5) for k, g in closed.groupby("month")},
    }


def run_replay(feats, meta, prices, qcfg, cal, months=24, embargo=5, trials_path=None, tag="replay"):
    """Validation (CPCV range, DSR) + day-by-day replay for one configuration."""
    last = pd.to_datetime(feats["date"]).dt.tz_localize(None).max().normalize()
    start = (last - pd.DateOffset(months=months)).normalize()
    print(f"[{tag}] window {start.date()} .. {last.date()}")

    # the comparison range: the same validation the daily run publishes. trials_path=None
    # means it is not logged (and the DSR is not deflated for other attempts).
    print(f"[{tag}] purged K-fold + CPCV on the full history")
    _, report = make_signals(feats, meta, qcfg, cal_name=cal, trials_path=trials_path)

    ev, _ = build_events(feats, qcfg)
    cand = candidates(feats, qcfg, start)
    prob = walk_forward_probs(ev, cand, qcfg, embargo)
    signals = build_signals(cand, prob, qcfg, cal)
    print(f"[{tag}] {len(cand)} primary candidates, {len(signals)} sized signals")

    ledger = replay_ledger(prices, signals, start, qcfg.get("cost_bps", 5))
    summary = summarize(ledger, prices, report, str(last.date()))
    summary["window"] = {"from": str(start.date()), "to": str(last.date()), "months": months,
                         "retrain": "monthly", "embargo_bdays": embargo}
    m = report["meta"]
    summary["backtest"] = {k: m.get(k) for k in ("dsr", "sr_per_trade", "sr_annualized", "expected_value",
                                                 "win_rate", "n_trades", "years")}
    summary["backtest"].update({"cpcv_path_sr": report["cpcv_path_sr"], "n_trials": report["n_trials"],
                                "stress_windows": report.get("stress_windows")})
    summary["diagnostics"] = compare(ledger, signals, ev, cand, prob, qcfg)

    ledger.to_csv(OUT / f"{tag}_trades.csv", index=False)
    signals.to_csv(OUT / f"{tag}_signals.csv", index=False)
    (OUT / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="stocks_etl/config.yaml")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--embargo", type=int, default=5, help="business days between training labels and trading")
    ap.add_argument("--refresh", action="store_true", help="re-download prices and recompute features")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    qcfg = dict(cfg.get("quant") or {}, mc_paths=1000)
    cal = cfg.get("calendar")
    OUT.mkdir(parents=True, exist_ok=True)

    prices = load_prices(cfg, a.refresh)
    feats, meta = load_features(prices, qcfg, a.refresh)
    summary = run_replay(feats, meta, prices, qcfg, cal, a.months, a.embargo)
    slim = {k: v for k, v in summary.items() if k not in ("equity_curve", "positions", "recent_closed")}
    print(json.dumps(slim, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Risk of the live paper book and diagnostics of the live sample (informational).

  - Exposure: gross, net and per bucket (sector; SPY/QQQ broad equity, GLD gold, TLT rates).
    Pending rows count: they fill at the next open.
  - Portfolio Expected Shortfall (99%, one day) with the correlations of MARKET-STRESS days,
    shrunk toward 1: results tested one by one say nothing about the day everything falls
    together (21 Oct video). Plus the worst historical day of today's book.
  - Edge decay: rolling live EV against the backtest EV +/- 2 standard errors.
  - Review horizon: P(cumulative result > 0) = Phi(mu sqrt(T) / sigma); the edge grows with T,
    the noise with sqrt(T) (1 Nov video).
  - Open-trade re-scoring: the 60% probability at entry is stale once the trade moved; how did
    labeled trades in the same (sessions elapsed, P&L) situation end? (22 Oct video).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from stocks_etl.risk import var_es_t

ETF_BUCKETS = {"SPY": "broad_equity", "QQQ": "broad_equity", "GLD": "gold", "TLT": "rates"}
ELAPSED_BINS = [(1, 5), (6, 10), (11, 15), (16, 10_000)]
PNL_EDGES = [-np.inf, -1.0, -0.5, -0.15, 0.15, 0.5, 1.0, np.inf]


def risk_bucket(ticker: str, sectors: Dict[str, str]) -> str:
    return ETF_BUCKETS.get(ticker) or sectors.get(ticker) or "other"


def stress_correlation(feats: pd.DataFrame, tickers: List[str], shrink: float = 0.2, min_days: int = 60) -> np.ndarray:
    """Correlation of daily returns on market-stress days, shrunk toward all-ones (in a crisis
    correlations rise); all-ones when there are too few stress days."""
    n = len(tickers)
    if n == 0:
        return np.zeros((0, 0))
    wide = feats.pivot(index="date", columns="ticker", values="log_ret_1").reindex(columns=tickers)
    mkt = feats.drop_duplicates("date").set_index("date")["mkt_regime"].reindex(wide.index)
    stress = wide[mkt == 2]
    if len(stress) < min_days:
        return np.ones((n, n))
    c = stress.corr().fillna(1.0).to_numpy()
    c = (1 - shrink) * c + shrink * np.ones((n, n))
    np.fill_diagonal(c, 1.0)
    return c


def book_risk(ledger: pd.DataFrame, signals: pd.DataFrame, feats: pd.DataFrame, sectors: Dict[str, str]) -> Dict:
    live = ledger[ledger["status"].isin(["open", "pending"])].copy()
    if live.empty:
        return {"positions": 0, "gross": 0.0, "net": 0.0, "by_bucket": {}}
    sig = signals.set_index("ticker")
    live["sigma"] = live["ticker"].map(sig["sigma_next"]).astype(float)
    live["nu"] = live["ticker"].map(sig["garch_nu"]).astype(float)
    live["w"] = live["size"].astype(float) * live["side"].astype(float)
    live["bucket"] = [risk_bucket(t, sectors) for t in live["ticker"]]
    tickers = list(live["ticker"])
    C = stress_correlation(feats, tickers)
    x = (live["w"] * live["sigma"]).fillna(0.0).to_numpy()
    sigma_p = float(math.sqrt(max(x @ C @ x, 0.0)))
    nu = float(np.nanmin(live["nu"])) if live["nu"].notna().any() else 5.0
    es_port = var_es_t(sigma_p, nu, 0.01)["es"] if sigma_p > 0 else 0.0
    es_each = [abs(w) * var_es_t(s, v if np.isfinite(v) else 5.0, 0.01)["es"]
               for w, s, v in zip(live["w"], live["sigma"], live["nu"]) if np.isfinite(s)]
    # worst historical day for each position and for today's book (same weights, past returns)
    wide = feats.pivot(index="date", columns="ticker", values="log_ret_1").reindex(columns=tickers)
    simple = np.expm1(wide)
    w = live.set_index("ticker")["w"]
    book_hist = (simple * w).sum(axis=1, min_count=1).dropna()
    worst = []
    for t, wt in w.items():
        s = simple[t].dropna()
        if len(s):
            d = (s * np.sign(wt)).idxmin()
            worst.append({"ticker": t, "date": str(pd.Timestamp(d).date()), "loss_pct_equity": float(abs(wt) * s[d] * np.sign(wt))})
    return {
        "positions": int(len(live)), "gross": float(live["w"].abs().sum()), "net": float(live["w"].sum()),
        "by_bucket": {b: float(g["w"].abs().sum()) for b, g in live.groupby("bucket")},
        "es99_1d_stressed_corr": float(es_port), "es99_1d_all_fall_together": float(sum(es_each)),
        "worst_day_book": ({"date": str(pd.Timestamp(book_hist.idxmin()).date()), "return": float(book_hist.min())}
                           if len(book_hist) else None),
        "worst_day_by_position": sorted(worst, key=lambda r: r["loss_pct_equity"])[:10],
    }


def edge_decay(closed_ret: pd.Series, bt_ev: Optional[float], bt_sd: Optional[float], window: int = 20) -> Dict:
    r = pd.Series(closed_ret, dtype=float).reset_index(drop=True)
    out = {"window": window, "backtest_ev": bt_ev, "points": []}
    if len(r) < window or bt_ev is None or not bt_sd:
        out["status"] = "collecting"
        return out
    roll = r.rolling(window).mean().dropna()
    band = 2 * bt_sd / math.sqrt(window)
    out.update(points=[{"trade": int(i + 1), "ev": float(v)} for i, v in roll.items()],
               band=[bt_ev - band, bt_ev + band],
               status="below_band" if roll.iloc[-1] < bt_ev - band else "inside_band")
    return out


def review_horizon(pnl: np.ndarray) -> Dict:
    pnl = np.asarray(pnl, float)
    if len(pnl) < 10 or pnl.std(ddof=1) == 0:
        return {}
    mu, sd = float(pnl.mean()), float(pnl.std(ddof=1))
    out = {"p_positive_after": {str(T): float(norm.cdf(mu * math.sqrt(T) / sd)) for T in (30, 100, 250, 500)}}
    out["trades_for_95pct"] = float((1.6449 * sd / mu) ** 2) if mu > 0 else None
    return out


def _bucket(elapsed: np.ndarray, u: np.ndarray):
    e = np.full(len(elapsed), -1)
    for i, (a, b) in enumerate(ELAPSED_BINS):
        e[(elapsed >= a) & (elapsed <= b)] = i
    return e, np.digitize(u, PNL_EDGES[1:-1])


def progress_table(ev: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Every labeled event, at every session it was open: (elapsed bucket, P&L bucket in units of
    its barrier width sigma_event) -> how it ended. Columns: e, u, n, p_tp, p_profit."""
    need = {"entry_px", "sigma_event", "start", "end", "side", "barrier", "ret_net", "ticker"}
    if ev is None or ev.empty or not need.issubset(ev.columns):
        return pd.DataFrame(columns=["e", "u", "n", "p_tp", "p_profit"])
    px = prices.copy()
    px["day"] = pd.to_datetime(px["date"])
    if px["day"].dt.tz is not None:
        px["day"] = px["day"].dt.tz_convert("UTC").dt.tz_localize(None)
    closes = {t: (pd.DatetimeIndex(g["day"].dt.normalize()), g["close"].to_numpy(float))
              for t, g in px.sort_values("day").groupby("ticker")}
    E, U, TP, PR = [], [], [], []
    for r in ev.itertuples():
        if r.barrier in ("entry_gap", "open") or not np.isfinite(r.entry_px) or not r.sigma_event:
            continue
        days, c = closes[r.ticker]
        i0 = days.searchsorted(pd.Timestamp(r.start).normalize(), side="right")
        i1 = days.searchsorted(pd.Timestamp(r.end).normalize(), side="left")
        if i1 <= i0:
            continue
        k = np.arange(1, i1 - i0 + 1)                         # sessions elapsed, still open at the close
        u = r.side * (c[i0:i1] / r.entry_px - 1) / r.sigma_event
        E.append(k)
        U.append(u)
        TP.append(np.full(len(k), r.barrier == "tp"))
        PR.append(np.full(len(k), r.ret_net > 0))
    if not E:
        return pd.DataFrame(columns=["e", "u", "n", "p_tp", "p_profit"])
    e, u = _bucket(np.concatenate(E), np.concatenate(U))
    df = pd.DataFrame({"e": e, "u": u, "tp": np.concatenate(TP), "pr": np.concatenate(PR)})
    df = df[df["e"] >= 0]
    g = df.groupby(["e", "u"])
    return pd.DataFrame({"n": g.size(), "p_tp": g["tp"].mean(), "p_profit": g["pr"].mean()}).reset_index()


def rescore_positions(ledger: pd.DataFrame, prices: pd.DataFrame, table: pd.DataFrame, k_tp: float,
                      k_sl: float, min_n: int = 30) -> List[Dict]:
    """Conditional odds for each open position from the progress table (informational)."""
    out = []
    if table is None or table.empty:
        return out
    px = prices.copy()
    px["day"] = pd.to_datetime(px["date"])
    if px["day"].dt.tz is not None:
        px["day"] = px["day"].dt.tz_convert("UTC").dt.tz_localize(None)
    lookup = table.set_index(["e", "u"])
    for r in ledger[ledger["status"] == "open"].itertuples():
        g = px[px["ticker"] == r.ticker].sort_values("day")
        if g.empty or not np.isfinite(r.entry_price):
            continue
        s, tp, sl = int(r.side), float(r.tp), float(r.sl)
        c0 = (k_sl * tp + k_tp * sl) / (k_tp + k_sl)              # signal close recovered from the barriers
        sig_e = s * (tp - sl) / (c0 * (k_tp + k_sl))
        elapsed = int((g["day"] >= pd.Timestamp(r.entry_date)).sum())
        u = s * (float(g["close"].iloc[-1]) / float(r.entry_price) - 1) / sig_e if sig_e else np.nan
        e_b, u_b = _bucket(np.array([elapsed]), np.array([u]))
        key = (int(e_b[0]), int(u_b[0]))
        row = lookup.loc[key] if key in lookup.index else None
        out.append({"ticker": r.ticker, "signal_date": str(r.signal_date), "sessions_elapsed": elapsed,
                    "pnl_sigma": float(u), "p_entry": float(r.p_success) if pd.notna(r.p_success) else None,
                    "n_similar": int(row["n"]) if row is not None else 0,
                    "p_tp_now": float(row["p_tp"]) if row is not None and row["n"] >= min_n else None,
                    "p_profit_now": float(row["p_profit"]) if row is not None and row["n"] >= min_n else None})
    return out

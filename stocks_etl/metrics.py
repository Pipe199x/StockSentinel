"""Performance metrics the advisor asks for, computed on a calendar-time portfolio.

Per-trade statistics (win rate, EV, SR per trade) ignore that positions overlap across
tickers. Here every trade is spread over the sessions it was open, marked to the close each
day, so the strategy gets a daily return series with drawdowns, Sortino, Calmar, CAGR and
time in market. Nothing in this module changes which trades are taken.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kurtosis, norm, skew

from stocks_etl.validation import deflated_sharpe, min_sample_size, sharpe

TRADING_DAYS = 252


def _naive_day(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return (t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t).normalize()


def _price_index(prices: pd.DataFrame):
    p = prices[["date", "ticker", "close"] + (["dividends"] if "dividends" in prices else [])].copy()
    p["day"] = pd.to_datetime(p["date"])
    if p["day"].dt.tz is not None:
        p["day"] = p["day"].dt.tz_convert("UTC").dt.tz_localize(None)
    p["day"] = p["day"].dt.normalize()
    if "dividends" not in p:
        p["dividends"] = 0.0
    out = {}
    for t, g in p.groupby("ticker"):
        g = g.sort_values("day")
        out[t] = (pd.DatetimeIndex(g["day"]), g["close"].to_numpy(float), g["dividends"].fillna(0).to_numpy(float))
    return out


def daily_pnl(trades: pd.DataFrame, prices: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Daily portfolio P&L (fraction of equity) from overlapping trades.

    Each trade enters at the open of the session after its signal (entry_px), is marked to
    every close, exits at exit_px, collects dividends on ex-dates in (entry, exit] and pays the
    round-trip cost on the exit day. A trade's daily P&L sums exactly to size * ret_net.
    Returns date, pnl, gross (sum of sizes), net (sum of size * side), n_open.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["date", "pnl", "gross", "net", "n_open"])
    idx = _price_index(prices)
    cost = 2 * float(cost_bps) / 1e4
    pnl, gross, net, n_open = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(int)
    for tr in trades.itertuples():
        days, close, div = idx[tr.ticker]
        i0 = int(days.searchsorted(_naive_day(tr.start), side="right"))      # entry session
        i1 = int(days.searchsorted(_naive_day(tr.end), side="left"))         # exit session
        e, x, w, s = float(tr.entry_px), float(tr.exit_px), float(tr.size), int(tr.side)
        if not (np.isfinite(e) and np.isfinite(x)) or e <= 0 or i0 >= len(days):
            continue
        i1 = max(min(i1, len(days) - 1), i0)
        prev = e
        for j in range(i0, i1 + 1):
            px = x if j == i1 else close[j]
            d = div[j] if j > i0 else 0.0
            day = days[j]
            pnl[day] += w * s * (px + d - prev) / e
            gross[day] += w
            net[day] += w * s
            n_open[day] += 1
            prev = px
        pnl[days[i1]] -= w * cost
    out = pd.DataFrame({"date": sorted(pnl)})
    out["pnl"] = [pnl[d] for d in out["date"]]
    out["gross"] = [gross[d] for d in out["date"]]
    out["net"] = [net[d] for d in out["date"]]
    out["n_open"] = [n_open[d] for d in out["date"]]
    return out


def calendar_returns(daily: pd.DataFrame, sessions: Sequence) -> pd.DataFrame:
    """daily_pnl on every session from the first entry to the last exit (flat days = 0)."""
    if daily.empty:
        return daily
    s = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(sessions).normalize())))
    s = s[(s >= daily["date"].min()) & (s <= daily["date"].max())]
    return (daily.set_index("date").reindex(s).fillna(0.0).rename_axis("date").reset_index())


def sortino(r: np.ndarray, target: float = 0.0) -> float:
    """Annualized Sortino. Downside deviation over ALL periods, days above target counting as 0
    (dropping the good days, as much software does, inflates it: 5 Nov video)."""
    r = np.asarray(r, float)
    dd = math.sqrt(np.mean(np.minimum(r - target, 0.0) ** 2)) if len(r) else 0.0
    return float((r.mean() - target) / dd * math.sqrt(TRADING_DAYS)) if dd > 0 else float("nan")


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, float)
    if not len(eq):
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(np.max(1 - eq / peak))


def calendar_stats(cal: pd.DataFrame, trades: pd.DataFrame, rf_daily: Optional[pd.Series] = None) -> Dict:
    """Calendar metrics of a daily return series (cal: date, pnl, gross, net)."""
    if cal is None or cal.empty:
        return {}
    r = cal["pnl"].to_numpy(float)
    eq = np.cumprod(1 + r)
    n = len(r)
    years = n / TRADING_DAYS
    cagr = float(eq[-1] ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else float("nan")
    mdd = max_drawdown(eq)
    sd = r.std(ddof=1) if n > 1 else 0.0
    sr = float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    rf = (rf_daily.reindex(pd.DatetimeIndex(cal["date"])).ffill().fillna(0.0).to_numpy()
          if rf_daily is not None and len(rf_daily) else np.zeros(n))
    ex = r - rf * cal["net"].clip(lower=0).to_numpy()        # long positions displace cash that would earn rf
    sd_ex = ex.std(ddof=1) if n > 1 else 0.0
    so = sortino(r)
    pnl = trades["pnl"].to_numpy(float) if trades is not None and len(trades) else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    ret = trades["ret_net"].to_numpy(float) if trades is not None and len(trades) else np.array([])
    rw, rl = ret[ret > 0], ret[ret < 0]
    log_eq = np.log(eq)
    t = np.arange(n)
    r2 = float(np.corrcoef(t, log_eq)[0, 1] ** 2) if n > 2 and np.std(log_eq) > 0 else float("nan")
    return {
        "sessions": int(n), "years": round(years, 2),
        "total_return": float(eq[-1] - 1), "cagr": cagr, "log_growth": float(np.mean(np.log1p(r)) * TRADING_DAYS),
        "volatility": float(sd * math.sqrt(TRADING_DAYS)),
        "sharpe": sr, "sharpe_excess_rf": float(ex.mean() / sd_ex * math.sqrt(TRADING_DAYS)) if sd_ex > 0 else float("nan"),
        "risk_free_used": bool(rf.any()),
        "sortino": so, "sortino_to_sharpe": float(so / sr) if sr and np.isfinite(sr) and sr > 0 else None,
        "max_drawdown": mdd, "calmar": float(cagr / mdd) if mdd > 0 and np.isfinite(cagr) else None,
        "time_in_market": float((cal["gross"] > 0).mean()),
        "avg_gross_exposure": float(cal["gross"].mean()), "max_gross_exposure": float(cal["gross"].max()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
        "payoff_ratio": float(rw.mean() / -rl.mean()) if len(rw) and len(rl) else None,
        "equity_r2": r2,
        "lag1_autocorr": float(pd.Series(r).autocorr(1)) if n > 3 else None,
    }


def equity_weekly(cal: pd.DataFrame) -> List[Dict]:
    """Weekly equity points for the dashboard chart."""
    if cal is None or cal.empty:
        return []
    eq = pd.Series(np.cumprod(1 + cal["pnl"].to_numpy(float)), index=pd.DatetimeIndex(cal["date"]))
    w = eq.resample("W-FRI").last().dropna()
    return [{"date": str(d.date()), "equity": round(float(v), 6)} for d, v in w.items()]


# --- effective sample size (AFML ch. 4) -----------------------------------------------------

def average_uniqueness(starts, ends, sessions) -> np.ndarray:
    """Average uniqueness of each interval [start, end] over a session grid: mean of
    1/concurrency across the sessions it spans. Overlapping trades share information."""
    s = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(sessions).normalize())))
    a = np.clip(s.searchsorted(pd.DatetimeIndex([_naive_day(x) for x in starts])), 0, len(s) - 1)
    b = np.clip(s.searchsorted(pd.DatetimeIndex([_naive_day(x) for x in ends])), 0, len(s) - 1)
    b = np.maximum(a, b)
    conc = np.zeros(len(s) + 1)
    np.add.at(conc, a, 1)
    np.add.at(conc, b + 1, -1)
    conc = np.cumsum(conc)[:-1]
    inv = np.concatenate([[0.0], np.cumsum(1.0 / np.maximum(conc, 1))])
    return (inv[b + 1] - inv[a]) / (b - a + 1)


def sr_ci(pnl: np.ndarray, level: float = 0.95) -> Dict:
    """Per-trade Sharpe with its standard error (skew/kurtosis aware) and a confidence interval."""
    pnl = np.asarray(pnl, float)
    n = len(pnl)
    if n < 3:
        return {"sr": None, "se": None, "ci": None}
    sr = sharpe(pnl)
    g3, g4 = float(skew(pnl)), float(kurtosis(pnl, fisher=False))
    se = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12) / (n - 1))
    z = norm.ppf(0.5 + level / 2)
    return {"sr": sr, "se": se, "ci": [sr - z * se, sr + z * se]}


def min_track_record(pnl: np.ndarray, sr_star: float = 0.0, alpha: float = 0.05) -> Optional[float]:
    """Minimum Track Record Length (Bailey & López de Prado): trades needed for PSR(sr*) >= 1 - alpha."""
    pnl = np.asarray(pnl, float)
    if len(pnl) < 4:
        return None
    sr = sharpe(pnl)
    if sr <= sr_star:
        return None
    g3, g4 = float(skew(pnl)), float(kurtosis(pnl, fisher=False))
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (norm.ppf(1 - alpha) / (sr - sr_star)) ** 2)


# --- losing streaks -------------------------------------------------------------------------

def p_run_at_least(L: int, n: int, q: float) -> float:
    """Exact P(longest run of losses >= L within n independent trades), loss probability q."""
    if L <= 0:
        return 1.0
    state = np.zeros(L)
    state[0] = 1.0
    hit = 0.0
    for _ in range(int(n)):
        new = np.zeros(L)
        new[0] = state.sum() * (1 - q)
        hit += state[L - 1] * q
        new[1:] = state[:-1] * q
        state = new
    return float(hit)


def streak_threshold(p_win: float, n: int = 30, alpha: float = 0.05) -> Optional[int]:
    """Smallest L with P(a run of >= L losses within n trades) < alpha (the calibrated breaker)."""
    q = 1 - float(p_win)
    if not 0 < q < 1:
        return None
    L = 1
    while p_run_at_least(L, n, q) >= alpha and L < 200:
        L += 1
    return L


# --- Gate 1: is the edge proven? (advisor: ~800 independent trades for 55% vs 50%) ----------

def gate1(bt: Dict, live_ret: np.ndarray, live_n_eff: float, trades_per_year: Optional[float],
          alpha: float = 0.05, power: float = 0.80) -> Dict:
    """Capital gate. Break-even win rate p0 = 1 / (1 + payoff) (not 0.5: payoffs are asymmetric);
    the backtest win rate p1 sets the sample a binomial test needs; the live (effective) trades
    are then tested against p0. 'pass' only with enough trades AND a significant live win rate."""
    w, aw, al = bt.get("win_rate"), bt.get("avg_win"), bt.get("avg_loss")
    out = {"status": "no_backtest_edge", "p_breakeven": None, "p_backtest": w, "trades_required": None,
           "live_trades": int(len(live_ret)), "live_effective_trades": float(live_n_eff),
           "years_to_go": None, "p_value": None}
    if not (w and aw and al) or al <= 0:
        return out
    p0 = 1.0 / (1.0 + aw / al)
    out["p_breakeven"] = p0
    if w <= p0 + 1e-6:
        return out
    req = min_sample_size(float(w), p0, alpha, power)
    out["trades_required"] = int(req)
    n = int(len(live_ret))
    if trades_per_year and trades_per_year > 0:
        out["years_to_go"] = max(req - live_n_eff, 0.0) / trades_per_year
    if live_n_eff < req:
        out["status"] = "collecting"
        return out
    k = int((np.asarray(live_ret) > 0).sum())
    out["p_value"] = float(binomtest(k, n, p0, alternative="greater").pvalue)
    out["status"] = "pass" if out["p_value"] < alpha else "fail"
    return out


# --- red flags (advisor's warning signs) ------------------------------------------------------

def red_flags(meta: Dict, cal_stats: Dict, report: Dict, garch: Dict[str, Dict], rcfg: Dict) -> List[Dict]:
    f = rcfg.get("flags", {}) or {}
    out: List[Dict] = []

    def add(code, severity, message, value=None, threshold=None):
        out.append({"code": code, "severity": severity, "message": message, "value": value, "threshold": threshold})

    sr = cal_stats.get("sharpe")
    if sr is not None and np.isfinite(sr):
        if sr > f.get("sharpe_high", 2.0):
            add("sharpe_high", "warning", "Calendar Sharpe above 2 is a classic sign of overfitting", sr, 2.0)
        elif sr < f.get("sharpe_low", 0.5):
            add("sharpe_low", "info", "Calendar Sharpe below 0.5: a weak edge, expect long drawdowns", sr, 0.5)
    wr = meta.get("win_rate")
    if wr is not None and wr > f.get("win_rate_high", 0.90):
        add("win_rate_high", "warning", "Win rate above 90%: the signature of grid / martingale payoffs", wr, 0.90)
    r2, ac = cal_stats.get("equity_r2"), cal_stats.get("lag1_autocorr")
    if (r2 is not None and np.isfinite(r2) and r2 > f.get("smooth_r2", 0.97) and (sr or 0) > 1.5) or \
            (ac is not None and np.isfinite(ac) and ac > 0.3):
        add("too_smooth", "warning", "The equity curve is suspiciously smooth (a perfect backtest is a red flag)", r2)
    if (meta.get("expected_value") or 0) <= 0:
        add("ev_nonpositive", "critical", "Expected value per trade after costs is not positive", meta.get("expected_value"), 0)
    ss = cal_stats.get("sortino_to_sharpe")
    if ss is not None and ss < f.get("sortino_sharpe_min", 1.5):
        add("sortino_sharpe_low", "info", "Sortino/Sharpe below 1.5: losses are relatively fat (left-tail risk)", ss, 1.5)
    if meta.get("dsr") is not None and meta.get("dsr") < report.get("dsr_threshold", 0.95):
        add("dsr_below_bar", "info", "Deflated Sharpe below 0.95: the edge is not statistically proven", meta.get("dsr"), 0.95)
    if report.get("regimes_undercovered"):
        add("regime_undercovered", "info", "Some traded regimes have fewer than 30 out-of-sample trades",
            report.get("regimes_undercovered"))
    for t, g in (garch or {}).items():
        if g.get("persistence") is not None and g["persistence"] >= 1:
            add("garch_nonstationary", "warning", f"{t}: GARCH persistence >= 1 (volatility does not mean-revert)",
                g["persistence"], 1.0)
        if g.get("nu") is not None and g["nu"] > f.get("nu_max", 7):
            add("tails_too_normal", "info", f"{t}: Student-t dof {g['nu']:.1f} > 7, tails close to normal", g["nu"], 7)
    fr = report.get("data_freshness") or {}
    if fr and not fr.get("news_fresh", True):
        add("news_stale", "warning", "News was not updated up to the signal day's cutoff", fr.get("news_fetched_utc"))
    dq = report.get("data_quality") or {}
    if dq.get("n_flags"):
        add("data_quality", "info", f"{dq['n_flags']} data-quality flags (see data_quality)", dq["n_flags"])
    return out

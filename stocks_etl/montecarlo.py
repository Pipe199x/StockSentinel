"""Strategy-level Monte Carlo (advisor, 17 Oct video: simulate 10,000 account paths before
risking a dollar; the key output is the chance of breaching your maximum-loss limit).

Two engines on the strategy's DAILY portfolio returns (not on a single asset's price):
  - stationary block bootstrap (Politis-Romano): resamples real days in blocks, so volatility
    clustering and fat tails come from the data itself;
  - Student-t: same mean and volatility, fat tails with the fitted degrees of freedom.
Plus the losing-streak distribution from resampled trade outcomes and a gambler's-ruin style
analytic approximation. Informational: nothing here changes a trade.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.stats import t as student_t

from stocks_etl.risk import max_losing_streak


def stationary_bootstrap(r: np.ndarray, horizon: int, n: int, block: float, seed: int) -> np.ndarray:
    """n x horizon resampled daily returns; blocks have geometric length with mean `block`."""
    r = np.asarray(r, float)
    rng = np.random.default_rng(seed)
    idx = np.empty((n, horizon), dtype=np.int64)
    idx[:, 0] = rng.integers(0, len(r), n)
    restart = rng.random((n, horizon)) < 1.0 / max(block, 1.0)
    fresh = rng.integers(0, len(r), (n, horizon))
    for j in range(1, horizon):
        idx[:, j] = np.where(restart[:, j], fresh[:, j], (idx[:, j - 1] + 1) % len(r))
    return r[idx]


def student_t_paths(r: np.ndarray, horizon: int, n: int, seed: int) -> np.ndarray:
    """Same mean and volatility as r, Student-t shocks (dof fitted, floored at 3)."""
    r = np.asarray(r, float)
    nu = max(float(student_t.fit(r, floc=r.mean())[0]), 3.0) if len(r) > 30 else 5.0
    scale = r.std(ddof=1) * math.sqrt((nu - 2) / nu)
    rng = np.random.default_rng(seed)
    return r.mean() + scale * rng.standard_t(nu, size=(n, horizon))


def _path_stats(paths: np.ndarray, max_loss: float, ruin_level: float) -> Dict:
    eq = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(eq), 1)), eq], axis=1), axis=1)[:, 1:]
    dd = np.max(1 - eq / peak, axis=1)
    final = eq[:, -1] - 1
    var95 = float(-np.percentile(final, 5))
    tail = final[final <= np.percentile(final, 5)]
    return {
        "p_profit": float((final > 0).mean()),
        "p_breach_max_loss": float((dd >= max_loss).mean()),
        "p_ruin": float((eq.min(axis=1) <= 1 - ruin_level).mean()),
        "drawdown_p50": float(np.percentile(dd, 50)), "drawdown_p95": float(np.percentile(dd, 95)),
        "drawdown_p99": float(np.percentile(dd, 99)),
        "return_p05": float(np.percentile(final, 5)), "return_p50": float(np.percentile(final, 50)),
        "return_p95": float(np.percentile(final, 95)),
        "var95": var95, "cvar95": float(-tail.mean()) if len(tail) else var95,
    }


def streak_distribution(trade_ret: np.ndarray, trades_per_year: float, n: int, seed: int) -> Dict:
    """Longest losing streak in a year of trades, resampling the real trade outcomes."""
    wins = np.asarray(trade_ret, float) > 0
    k = max(int(round(trades_per_year)), 1)
    if len(wins) < 10:
        return {}
    rng = np.random.default_rng(seed)
    longest = np.array([max_losing_streak(wins[rng.integers(0, len(wins), k)]) for _ in range(n)])
    return {"trades_per_year": k, "p50": float(np.percentile(longest, 50)),
            "p95": float(np.percentile(longest, 95)), "max": int(longest.max())}


def analytic_ruin(pnl: np.ndarray, ruin_level: float) -> float:
    """P(ever losing `ruin_level` of equity) for a random walk with the trades' drift and
    volatility: exp(-2 mu D / sigma^2) when the drift is positive, 1 otherwise."""
    pnl = np.asarray(pnl, float)
    if len(pnl) < 2:
        return float("nan")
    mu, var = pnl.mean(), pnl.var(ddof=1)
    if mu <= 0 or var <= 0:
        return 1.0
    return float(min(1.0, math.exp(-2 * mu * ruin_level / var)))


def strategy_mc(daily_r: np.ndarray, trades, rcfg: Dict, seed: int = 42) -> Dict:
    """Monte Carlo of the strategy's next year (rcfg: max_loss_limit, ruin_level,
    strategy_mc_paths, mc_block, mc_horizon_days)."""
    r = np.asarray(daily_r, float)
    if len(r) < 60:
        return {"status": "insufficient_data", "sessions": int(len(r))}
    horizon = int(rcfg.get("mc_horizon_days", 252))
    n = int(rcfg.get("strategy_mc_paths", 5000))
    max_loss = float(rcfg.get("max_loss_limit", 0.10))
    ruin = float(rcfg.get("ruin_level", 0.30))
    years = len(r) / 252
    tpy = len(trades) / years if trades is not None and len(trades) and years > 0 else 0
    pnl = trades["pnl"].to_numpy(float) if trades is not None and len(trades) else np.array([])
    ret = trades["ret_net"].to_numpy(float) if trades is not None and len(trades) else np.array([])
    return {
        "status": "ok", "paths": n, "horizon_days": horizon, "max_loss_limit": max_loss, "ruin_level": ruin,
        "bootstrap": _path_stats(stationary_bootstrap(r, horizon, n, float(rcfg.get("mc_block", 20)), seed),
                                 max_loss, ruin),
        "student_t": _path_stats(student_t_paths(r, horizon, n, seed + 1), max_loss, ruin),
        "losing_streak_1y": streak_distribution(ret, tpy, 2000, seed + 2),
        "analytic_ruin": analytic_ruin(pnl, ruin),
    }

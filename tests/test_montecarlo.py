import numpy as np
import pytest
import pandas as pd

from stocks_etl.montecarlo import analytic_ruin, stationary_bootstrap, strategy_mc


def _daily(seed=0, n=1500):
    return np.random.default_rng(seed).standard_t(4, n) * 0.004 + 0.0003


def _trades(n=400, seed=1):
    r = np.random.default_rng(seed).normal(0.002, 0.03, n)
    return pd.DataFrame({"ret_net": r, "pnl": 0.05 * r})


def test_bootstrap_resamples_real_days_in_blocks():
    r = np.arange(100, dtype=float)
    p = stationary_bootstrap(r, 50, 20, block=10, seed=0)
    assert p.shape == (20, 50) and set(np.unique(p)) <= set(r)
    steps = np.diff(p, axis=1)
    assert (steps == 1).mean() > 0.8          # mostly consecutive days (blocks), with random restarts


def test_strategy_mc_is_seeded_and_breach_probability_falls_with_a_looser_limit():
    cfg = {"strategy_mc_paths": 2000, "mc_horizon_days": 252, "mc_block": 20, "ruin_level": 0.3}
    a = strategy_mc(_daily(), _trades(), {**cfg, "max_loss_limit": 0.02}, seed=1)
    b = strategy_mc(_daily(), _trades(), {**cfg, "max_loss_limit": 0.02}, seed=1)
    c = strategy_mc(_daily(), _trades(), {**cfg, "max_loss_limit": 0.10}, seed=1)
    assert a == b
    assert a["bootstrap"]["p_breach_max_loss"] >= c["bootstrap"]["p_breach_max_loss"]
    assert 0 <= a["student_t"]["p_profit"] <= 1 and a["losing_streak_1y"]["p95"] >= a["losing_streak_1y"]["p50"]


def test_analytic_ruin():
    assert analytic_ruin(np.array([-0.01, 0.0, -0.02, 0.005]), 0.3) == 1.0      # no positive drift: ruin
    pnl = np.random.default_rng(0).normal(0.01, 0.02, 500)
    expected = np.exp(-2 * pnl.mean() * 0.3 / pnl.var(ddof=1))       # gambler's-ruin approximation
    assert analytic_ruin(pnl, 0.3) == pytest.approx(expected) and expected < 1e-4


def test_insufficient_history():
    assert strategy_mc(np.zeros(10), _trades(), {})["status"] == "insufficient_data"

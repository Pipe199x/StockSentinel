import numpy as np
import pandas as pd
import pytest

from stocks_etl.labeling import triple_barrier_live
from stocks_etl.paper import load_ledger, update_ledger
from stocks_etl.primary import primary_side


def _ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = c * np.exp(rng.normal(0, 0.006, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.008, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.008, n)))
    return o, h, l, c


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_live_barrier_matches_paper_ledger(seed):
    """The backtest label must equal what the live paper bot books for the same signal."""
    o, h, l, c = _ohlc(seed=seed)
    days = pd.bdate_range("2025-01-02", periods=len(c), tz="UTC")
    prices = pd.DataFrame({"date": days, "ticker": "AAA", "open": o, "high": h, "low": l, "close": c})
    rng = np.random.default_rng(seed + 10)
    pos = np.arange(5, 270, 13)
    side = rng.choice([-1, 1], len(pos))
    sigma = np.full(len(pos), 0.03)
    tmax = rng.integers(3, 15, len(pos))
    tb = triple_barrier_live(o, h, l, c, pos, side, sigma, 1.5, 1.5, tmax)
    for i, p0 in enumerate(pos):
        tp, sl = c[p0] * (1 + side[i] * 1.5 * 0.03), c[p0] * (1 - side[i] * 1.5 * 0.03)
        sig = pd.DataFrame([{"ticker": "AAA", "as_of": str(days[p0].date()), "side": int(side[i]),
                             "kelly_size": 0.05, "p_success": 0.6, "tp": tp, "sl": sl,
                             "t_max_date": str(days[p0 + tmax[i]].date())}])
        led = update_ledger(update_ledger(load_ledger("none"), prices, sig, 0), prices, sig, 0)
        r = led.iloc[0]
        assert r.status == "closed"
        assert tb["ret"].iloc[i] == pytest.approx(float(r.ret_gross), abs=1e-12)
        assert days[tb["end_pos"].iloc[i]].date().isoformat() == r.exit_date
        expect = {"time": "t"}.get(r.exit_reason, r.exit_reason)
        assert tb["barrier"].iloc[i] == expect


def test_strategy_filter_turns_off_mean_reversion():
    f = pd.DataFrame({"hurst": [0.6, 0.3], "zscore": [1.0, 2.0], "half_life": [5, 5], "log_ret_20": [0.05, 0.01],
                      "sigma_garch": [0.01, 0.01], "regime": [0, 0]})
    both = primary_side(f, {})
    mom = primary_side(f, {"strategies": ["momentum"]})
    assert list(both["strategy"]) == [1, 2]
    assert list(mom["strategy"]) == [1, 0] and mom["side"].iloc[1] == 0

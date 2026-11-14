import itertools
import math

import numpy as np
import pandas as pd
import pytest

from stocks_etl.metrics import (average_uniqueness, daily_pnl, gate1, max_drawdown, p_run_at_least, red_flags,
                                sortino, streak_threshold)


def test_sortino_counts_good_days_as_zero_in_the_downside_deviation():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.0])
    dd = math.sqrt(np.mean(np.minimum(r, 0) ** 2))            # all 5 periods, positives as 0
    assert sortino(r) == pytest.approx(r.mean() / dd * math.sqrt(252))
    wrong = r.mean() / r[r < 0].std(ddof=0) * math.sqrt(252)  # the common mistake inflates it
    assert sortino(r) < wrong


def test_max_drawdown():
    assert max_drawdown(np.array([1.1, 0.99, 1.2, 0.9])) == pytest.approx(1 - 0.9 / 1.2)


def test_daily_pnl_sums_to_size_times_net_return_with_costs_and_dividends():
    days = pd.bdate_range("2026-01-05", periods=8, tz="UTC")
    close = np.array([100, 101, 103, 102, 104, 105, 104, 106], float)
    div = np.zeros(8)
    div[3] = 0.8
    prices = pd.DataFrame({"date": days, "ticker": "AAA", "close": close, "dividends": div})
    entry, exit_ = 101.5, 104.5          # enters at the open of day 1, exits intraday on day 5
    ret_net = 1 * ((exit_ + 0.8) / entry - 1) - 0.001
    trades = pd.DataFrame([{"ticker": "AAA", "start": days[0].tz_localize(None), "end": days[5].tz_localize(None),
                            "side": 1, "size": 0.05, "entry_px": entry, "exit_px": exit_, "ret_net": ret_net,
                            "pnl": 0.05 * ret_net}])
    d = daily_pnl(trades, prices, cost_bps=5)
    assert len(d) == 5 and d["pnl"].sum() == pytest.approx(0.05 * ret_net, abs=1e-12)
    assert (d["gross"] == 0.05).all()


def test_average_uniqueness_afml_example():
    # AFML ch.4: three labels spanning t0-t2, t2-t3, t4-t5 on sessions t0..t5
    s = pd.date_range("2026-01-01", periods=6, freq="D")
    u = average_uniqueness([s[0], s[2], s[4]], [s[2], s[3], s[5]], s)
    assert u == pytest.approx([(1 + 1 + 0.5) / 3, (0.5 + 1) / 2, 1.0])


def _brute_run(L, n, q):
    p = 0.0
    for seq in itertools.product([0, 1], repeat=n):           # 1 = loss
        run = best = 0
        for x in seq:
            run = run + 1 if x else 0
            best = max(best, run)
        k = sum(seq)
        p += (best >= L) * q ** k * (1 - q) ** (n - k)
    return p


@pytest.mark.parametrize("L,n,q", [(2, 6, 0.5), (3, 10, 0.48), (4, 12, 0.6)])
def test_p_run_at_least_matches_brute_force(L, n, q):
    assert p_run_at_least(L, n, q) == pytest.approx(_brute_run(L, n, q), abs=1e-12)


def test_streak_threshold_values_quoted_to_the_user():
    assert streak_threshold(0.52, 30) == 8 and streak_threshold(0.40, 30) == 11


def test_gate1_uses_the_break_even_win_rate_not_one_half():
    bt = {"win_rate": 0.40, "avg_win": 0.02, "avg_loss": 0.01}     # 2:1 payoff: break-even 1/3
    g = gate1(bt, np.array([]), 0.0, 70.0)
    assert g["p_breakeven"] == pytest.approx(1 / 3) and g["status"] == "collecting" and g["trades_required"] > 0
    assert gate1({"win_rate": 0.30, "avg_win": 0.02, "avg_loss": 0.01}, np.array([]), 0, 70)["status"] == "no_backtest_edge"
    rng = np.random.default_rng(0)
    wins = rng.random(5000) < 0.40
    g = gate1(bt, np.where(wins, 0.02, -0.01), 5000.0, 70.0)
    assert g["status"] == "pass" and g["p_value"] < 0.05


def test_red_flags_catch_the_advisors_warning_signs():
    flags = red_flags({"win_rate": 0.97, "expected_value": -0.001, "dsr": 0.5},
                      {"sharpe": 2.6, "equity_r2": 0.99, "sortino_to_sharpe": 1.1}, {"dsr_threshold": 0.95},
                      {"AAA": {"persistence": 1.01, "nu": 12.0}}, {})
    codes = {f["code"] for f in flags}
    assert {"sharpe_high", "win_rate_high", "too_smooth", "ev_nonpositive", "sortino_sharpe_low", "dsr_below_bar",
            "garch_nonstationary", "tails_too_normal"} <= codes

"""Cash dividends: split-adjusted, credited to longs and debited to shorts for ex-dates in
(entry day, exit day], identically in the backtest labels and the paper ledger."""
import numpy as np
import pandas as pd
import pytest

from etl_common.corporate_actions import attach_dividends, refresh_dividends, split_adjusted_cash
from stocks_etl.labeling import triple_barrier_live
from stocks_etl.paper import load_ledger, rebase_for_splits, update_ledger


def test_split_adjusted_cash_divides_by_later_splits_only():
    divs = pd.DataFrame({"ticker": ["NVDA", "NVDA", "NVDA"], "ex_date": ["2021-06-09", "2023-06-07", "2024-09-12"],
                         "pay_date": None, "cash_amount": [0.16, 0.04, 0.01], "dividend_type": "CD",
                         "frequency": 4, "id": ["a", "b", "c"]})
    splits = pd.DataFrame({"ticker": ["NVDA", "NVDA"], "execution_date": ["2021-07-20", "2024-06-10"],
                           "split_from": [1.0, 1.0], "split_to": [4.0, 10.0]})
    out = split_adjusted_cash(divs, splits)
    assert out["cash_adj"].tolist() == pytest.approx([0.16 / 40, 0.04 / 10, 0.01])


def test_attach_dividends_on_ex_date_rows():
    days = pd.bdate_range("2026-01-05", periods=4, tz="UTC")
    px = pd.DataFrame({"date": days, "ticker": "AAA", "close": 100.0})
    divs = pd.DataFrame({"ticker": ["AAA"], "ex_date": ["2026-01-07"], "cash_adj": [0.5]})
    assert attach_dividends(px, divs)["dividends"].tolist() == [0.0, 0.0, 0.5, 0.0]


def _flat(n=8):
    o = h = l = c = np.full(n, 100.0)
    return o, h * 1.001, l * 0.999, c


@pytest.mark.parametrize("side", [1, -1])
def test_ex_date_on_entry_day_does_not_pay_on_exit_day_does(side):
    o, h, l, c = _flat()
    div = np.zeros(8)
    div[1] = 1.0      # ex-date = entry day (signal at 0, entry at the open of 1): not earned
    div[3] = 2.0      # ex-date = exit day (time barrier at 0 + 3): earned / paid
    tb = triple_barrier_live(o, h, l, c, np.array([0]), np.array([side]), np.array([0.05]), 1.5, 1.5,
                             np.array([3]), div=div)
    assert tb["barrier"].iloc[0] == "t" and tb["div"].iloc[0] == 2.0
    assert tb["ret"].iloc[0] == pytest.approx(side * ((100.0 + 2.0) / 100.0 - 1))   # shorts pay it


def _ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = c * np.exp(rng.normal(0, 0.006, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.008, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.008, n)))
    return o, h, l, c


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_live_labels_match_paper_ledger_with_dividends(seed):
    o, h, l, c = _ohlc(seed=seed)
    rng = np.random.default_rng(seed + 20)
    div = np.where(rng.random(len(c)) < 0.05, rng.uniform(0.2, 1.5, len(c)), 0.0)
    days = pd.bdate_range("2025-01-02", periods=len(c), tz="UTC")
    prices = pd.DataFrame({"date": days, "ticker": "AAA", "open": o, "high": h, "low": l, "close": c,
                           "dividends": div})
    pos = np.arange(5, 270, 11)
    side = rng.choice([-1, 1], len(pos))
    tmax = rng.integers(3, 15, len(pos))
    tb = triple_barrier_live(o, h, l, c, pos, side, np.full(len(pos), 0.03), 1.5, 1.5, tmax, div=div)
    paid = 0
    for i, p0 in enumerate(pos):
        tp, sl = c[p0] * (1 + side[i] * 1.5 * 0.03), c[p0] * (1 - side[i] * 1.5 * 0.03)
        sig = pd.DataFrame([{"ticker": "AAA", "as_of": str(days[p0].date()), "side": int(side[i]),
                             "kelly_size": 0.05, "p_success": 0.6, "tp": tp, "sl": sl,
                             "t_max_date": str(days[p0 + tmax[i]].date())}])
        r = update_ledger(update_ledger(load_ledger("none"), prices, sig, 0), prices, sig, 0).iloc[0]
        assert r.status == "closed"
        assert tb["ret"].iloc[i] == pytest.approx(float(r.ret_gross), abs=1e-12)
        assert tb["div"].iloc[i] == pytest.approx(float(r.div_cash), abs=1e-12)
        paid += tb["div"].iloc[i] > 0
    assert paid > 0          # the seeds do exercise trades that hold through an ex-date


def test_split_during_a_position_rescales_stored_prices():
    days = pd.bdate_range("2026-01-05", periods=5, tz="UTC")
    pre = pd.DataFrame({"date": days, "ticker": "AAA", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    sig = pd.DataFrame([{"ticker": "AAA", "as_of": str(days[0].date()), "side": 1, "kelly_size": 0.05,
                         "p_success": 0.6, "tp": 110.0, "sl": 95.0, "t_max_date": str(days[4].date())}])
    led = update_ledger(update_ledger(load_ledger("none"), pre.iloc[:1], sig, 5), pre.iloc[:3], sig, 5)
    assert led.iloc[0].status == "open" and led.iloc[0].entry_price == 100.0
    post = pre.assign(**{k: pre[k] / 10 for k in ("open", "high", "low", "close")})   # history re-adjusted 1->10
    led2, rebased = rebase_for_splits(led, post)
    r = led2.iloc[0]
    assert len(rebased) == 1 and rebased[0]["factor"] == pytest.approx(0.1)
    assert (r.entry_price, r.tp, r.sl) == pytest.approx((10.0, 11.0, 9.5))
    assert update_ledger(led2, post.iloc[:3], sig, 5).iloc[0].status == "open"     # no phantom stop-out
    phantom = update_ledger(led, post.iloc[:3], sig, 5).iloc[0]                     # without the rebase
    assert (phantom.status, phantom.exit_reason) == ("closed", "sl")          # a phantom stop-loss


class _Refused:
    def dividends(self, *a):
        raise SystemExit("Massive refused the request (HTTP 403)")

    def splits(self, *a):
        return iter(())


def test_refresh_falls_back_to_last_table_when_the_api_fails(tmp_path):
    path = tmp_path / "dividends.csv"
    pd.DataFrame({"ticker": ["AAA"], "ex_date": ["2026-01-06"], "pay_date": [None], "cash_amount": [0.5],
                  "split_factor": [1.0], "cash_adj": [0.5], "dividend_type": ["CD"], "frequency": [4],
                  "id": ["x"]}).to_csv(path, index=False)
    px = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=3, tz="UTC"), "ticker": "AAA", "close": 1.0})
    out, status = refresh_dividends(px, ["AAA"], str(path), client=_Refused())
    assert status["source"] == "last_published" and "403" in status["error"]
    assert out["dividends"].tolist() == [0.0, 0.5, 0.0]

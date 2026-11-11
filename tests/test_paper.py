import pandas as pd
import pytest

from stocks_etl.paper import load_ledger, summarize, update_ledger

DAYS = pd.bdate_range("2026-01-05", periods=8, tz="UTC")


def _px(rows):
    """rows: list of (open, high, low, close) for ticker AAA over DAYS."""
    return pd.DataFrame([{"date": d, "ticker": "AAA", "open": o, "high": h, "low": l, "close": c}
                         for d, (o, h, l, c) in zip(DAYS, rows)])


def _sig(day=0, side=1, size=0.05, tp=110.0, sl=95.0, exit_by=6):
    return pd.DataFrame([{"ticker": "AAA", "as_of": str(DAYS[day].date()), "side": side, "kelly_size": size,
                          "p_success": 0.6, "tp": tp, "sl": sl, "t_max_date": str(DAYS[exit_by].date())}])


FLAT = [(100, 101, 99, 100)] * 8


def test_signal_becomes_pending_then_fills_at_next_open():
    led = update_ledger(load_ledger("missing.csv"), _px(FLAT[:1]), _sig(), 5)
    assert list(led["status"]) == ["pending"]
    rows = list(FLAT)
    rows[1] = (102, 103, 101, 102)
    led = update_ledger(led, _px(rows[:2]), _sig(), 5)
    r = led.iloc[0]
    assert r.status == "open" and r.entry_price == 102 and r.entry_date == str(DAYS[1].date())


def test_take_profit_intraday_and_stop_on_double_touch():
    rows = list(FLAT)
    rows[2] = (104, 111, 103, 105)            # touches TP 110 intraday
    led = update_ledger(update_ledger(load_ledger("x"), _px(rows[:1]), _sig(), 5), _px(rows), _sig(), 5)
    r = led.iloc[0]
    assert (r.status, r.exit_reason, r.exit_price) == ("closed", "tp", 110.0)
    assert r.ret_net == pytest.approx(110 / 100 - 1 - 0.001)
    rows[2] = (100, 111, 94, 100)             # touches both: stop wins
    led = update_ledger(update_ledger(load_ledger("x"), _px(rows[:1]), _sig(), 5), _px(rows), _sig(), 5)
    assert led.iloc[0].exit_reason == "sl" and led.iloc[0].exit_price == 95.0


def test_gap_through_stop_fills_at_open_and_time_exit():
    rows = list(FLAT)
    rows[3] = (90, 91, 89, 90)
    led = update_ledger(update_ledger(load_ledger("x"), _px(rows[:1]), _sig(), 5), _px(rows), _sig(), 5)
    assert (led.iloc[0].exit_reason, led.iloc[0].exit_price) == ("sl", 90.0)
    sig = _sig(exit_by=4)
    led = update_ledger(update_ledger(load_ledger("x"), _px(FLAT[:1]), sig, 5), _px(FLAT), sig, 5)
    assert led.iloc[0].exit_reason == "time" and led.iloc[0].exit_date == str(DAYS[4].date())


def test_short_side_and_entry_gap():
    rows = list(FLAT)
    rows[1] = (94, 95, 93, 94)                # short entry opens below its TP already
    led = update_ledger(update_ledger(load_ledger("x"), _px(rows[:1]), _sig(side=-1, tp=95, sl=105), 5),
                        _px(rows[:2]), _sig(side=-1, tp=95, sl=105), 5)
    r = led.iloc[0]
    assert r.exit_reason == "entry_gap" and r.ret_gross == 0


def test_rerun_is_idempotent_and_one_position_per_ticker():
    px = _px(FLAT[:3])
    led = update_ledger(load_ledger("x"), px, _sig(day=0), 5)
    led = update_ledger(led, px, _sig(day=0), 5)
    assert len(led) == 1
    led = update_ledger(led, px, _sig(day=2), 5)   # still open: new signal on same ticker is skipped
    assert len(led) == 1


def test_summary_gate_collecting_and_equity():
    rows = list(FLAT)
    rows[2] = (104, 111, 103, 105)
    led = update_ledger(update_ledger(load_ledger("x"), _px(rows[:1]), _sig(), 5), _px(rows), _sig(), 5)
    s = summarize(led, _px(rows), {"cpcv_path_sr": [0.01, 0.03], "meta": {"sr_per_trade": 0.04}}, "2026-01-14")
    assert s["gate0"]["status"] == "collecting" and s["counts"]["closed"] == 1
    assert s["total_return"] == pytest.approx(0.05 * (110 / 100 - 1 - 0.001))

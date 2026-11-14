import numpy as np
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


def test_open_position_rewalks_from_entry_after_partial_bar():
    # 2026-11-13 incident: a session first seen as a partial bar (no touch yet) must be
    # re-checked once its full bar exists, not skipped because last_checked already covers it.
    partial = FLAT[:2] + [(100, 101, 99.5, 100.5)]
    led = update_ledger(load_ledger("x"), _px(FLAT[:1]), _sig(), 5)        # pending
    led = update_ledger(led, _px(partial), _sig(), 5)                      # open, checked on a partial day 2
    assert led.iloc[0].status == "open" and led.iloc[0].last_checked == str(DAYS[2].date())
    full = FLAT[:2] + [(100, 111, 99.5, 108)] + FLAT[3:]                   # the full day 2 touched TP
    led = update_ledger(led, _px(full), _sig(), 5)
    r = led.iloc[0]
    assert (r.status, r.exit_reason, r.exit_date, r.exit_price) == ("closed", "tp", str(DAYS[2].date()), 110.0)


def test_closed_row_is_revised_and_reported_when_bars_change():
    sig = _sig(exit_by=4)
    led = update_ledger(update_ledger(load_ledger("x"), _px(FLAT[:1]), sig, 5), _px(FLAT), sig, 5)
    assert led.iloc[0].exit_reason == "time"
    revised = FLAT[:3] + [(99, 100, 94, 96)] + FLAT[4:]                    # corrected bar: day 3 hit the stop
    revisions = []
    led = update_ledger(led, _px(revised), sig, 5, revisions=revisions)
    assert (led.iloc[0].exit_reason, led.iloc[0].exit_date) == ("sl", str(DAYS[3].date()))
    assert len(revisions) == 1 and revisions[0]["old"]["exit_reason"] == "time"
    # unchanged bars: closed rows stay as they are and nothing is reported
    again = []
    update_ledger(led, _px(revised), sig, 5, revisions=again)
    assert again == []


def test_signal_published_after_the_open_is_missed_not_filled():
    # signal on Mon 2026-01-05; the next session opens Tue 14:30 UTC
    late = update_ledger(load_ledger("x"), _px(FLAT[:1]), _sig(), 5, published_utc="2026-01-06T15:00:00+00:00")
    assert (late.iloc[0].status, late.iloc[0].exit_reason) == ("missed", "late_publish")
    led = update_ledger(late, _px(FLAT[:3]), _sig(day=2), 5, published_utc="2026-01-08T02:00:00+00:00")
    assert list(led["status"]) == ["missed", "pending"]          # a missed row never blocks the ticker
    ok = update_ledger(load_ledger("x"), _px(FLAT[:1]), _sig(), 5, published_utc="2026-01-06T02:00:00+00:00",
                       config_hash="h1", strategy_label="V4")
    assert (ok.iloc[0].status, ok.iloc[0].config_hash, ok.iloc[0].strategy_label) == ("pending", "h1", "V4")


def test_legacy_rows_are_tagged_once(tmp_path):
    path = tmp_path / "ledger.csv"
    pd.DataFrame({"ticker": ["HD", "TLT"], "signal_date": ["2026-11-11", "2026-11-12"], "side": [1, -1],
                  "size": [0.03, 0.001], "status": ["open", "open"]}).to_csv(path, index=False)
    led = load_ledger(str(path))
    assert list(led["config_hash"]) == ["f71882e35e28", "c3f2b4fc9bfd"]
    assert list(led["strategy_label"]) == ["V4", "V4+news"]
    led.to_csv(path, index=False)
    assert list(load_ledger(str(path))["config_hash"]) == ["f71882e35e28", "c3f2b4fc9bfd"]   # idempotent


def _closed(rets, h="h1"):
    return pd.DataFrame([{"ticker": "AAA", "signal_date": str(DAYS[0].date()), "side": 1, "size": 0.05,
                          "status": "closed", "entry_date": str(DAYS[1].date()), "entry_price": 100.0,
                          "exit_date": str(DAYS[2].date()), "exit_price": 100.0, "exit_reason": "time",
                          "ret_net": r, "config_hash": h, "strategy_label": h} for r in rets])


def _gate(led, cpcv=(0.05, 0.06, 0.07)):
    rep = {"config_hash": "h1", "cpcv_path_sr": list(cpcv), "meta": {"sr_per_trade": 0.06}}
    return summarize(led, _px(FLAT), rep, "2026-01-14")


def test_gate0_states_and_current_config_only():
    rng = np.random.default_rng(0)
    good = list(0.004 + 0.02 * rng.standard_normal(40))
    s = _gate(_closed(good[:10]))
    assert s["gate0"]["status"] == "collecting" and s["gate0"]["n_closed"] == 10
    s = _gate(_closed(good))
    assert s["gate0"]["status"] == "pass" and s["gate0"]["min_detectable_gap"] > 0
    flat_neg = [x - np.mean(good) - 0.0001 for x in good]                 # EV slightly negative, noisy
    assert _gate(_closed(flat_neg))["gate0"]["status"] == "inconclusive"
    bad = list(-0.02 + 0.005 * rng.standard_normal(40))                   # far below a strong backtest
    assert _gate(_closed(bad), cpcv=(0.5, 0.6, 0.7))["gate0"]["status"] == "fail"
    # trades of another configuration never count for the current one
    s = _gate(pd.concat([_closed(good[:10]), _closed(bad, h="old")], ignore_index=True))
    assert s["gate0"]["status"] == "collecting" and s["current_closed"] == 10 and s["counts"]["closed"] == 50
    assert {r["config_hash"] for r in s["by_config"]} == {"h1", "old"}


def test_summary_gate_collecting_and_equity():
    rows = list(FLAT)
    rows[2] = (104, 111, 103, 105)
    led = update_ledger(update_ledger(load_ledger("x"), _px(rows[:1]), _sig(), 5), _px(rows), _sig(), 5)
    s = summarize(led, _px(rows), {"cpcv_path_sr": [0.01, 0.03], "meta": {"sr_per_trade": 0.04}}, "2026-01-14")
    assert s["gate0"]["status"] == "collecting" and s["counts"]["closed"] == 1
    assert s["total_return"] == pytest.approx(0.05 * (110 / 100 - 1 - 0.001))

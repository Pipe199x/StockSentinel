"""Price data-quality gate: fatal problems stop the run, anomalies are only flagged."""
import numpy as np
import pandas as pd

from etl_common.validation import validate_prices

NOW = "2026-11-14T02:00:00Z"                       # last completed session: Friday 2026-11-13
DAYS = pd.bdate_range("2026-09-01", "2026-11-13")


def _frame(tickers=("SPY", "AAA", "BBB", "CCC", "DDD", "EEE"), days=DAYS):
    rows = []
    for k, t in enumerate(tickers):
        px = 100 + k + np.cumsum(np.sin(np.arange(len(days))) * 0.5)
        for d, c in zip(days, px):
            rows.append({"date": pd.Timestamp(d, tz="UTC"), "ticker": t, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": 1e6})
    return pd.DataFrame(rows)


def test_clean_data_passes():
    _, rep = validate_prices(_frame(), "NYSE", NOW, "SPY")
    assert rep.fatal == [] and rep.stale == [] and rep.last_session == "2026-11-13"


def test_duplicates_and_bad_last_bar_are_fatal():
    df = _frame()
    _, rep = validate_prices(pd.concat([df, df.tail(1)]), "NYSE", NOW, "SPY")
    assert any("duplicated" in f for f in rep.fatal)
    df.loc[df.index[-1], "close"] = -1.0
    _, rep = validate_prices(df, "NYSE", NOW, "SPY")
    assert any("non-positive" in f for f in rep.fatal)


def test_missing_last_session_benchmark_is_fatal_other_ticker_is_stale():
    df = _frame()
    last = df["date"] == df["date"].max()
    _, rep = validate_prices(df[~(last & (df["ticker"] == "AAA"))], "NYSE", NOW, "SPY")
    assert rep.fatal == [] and rep.stale == ["AAA"]
    _, rep = validate_prices(df[~(last & (df["ticker"] == "SPY"))], "NYSE", NOW, "SPY")
    assert any("benchmark SPY" in f for f in rep.fatal)


def test_unadjusted_split_is_fatal_but_a_crash_is_only_flagged():
    df = _frame()
    i = df.index[(df["ticker"] == "AAA") & (df["date"] == pd.Timestamp("2026-10-01", tz="UTC"))][0]
    after = (df["ticker"] == "AAA") & (df.index >= i)
    split = df.copy()
    split.loc[after, ["open", "high", "low", "close"]] /= 2.0          # a 2:1 split left unadjusted
    _, rep = validate_prices(split, "NYSE", NOW, "SPY")
    assert any("unadjusted split" in f for f in rep.fatal)
    crash = df.copy()
    crash.loc[after, ["open", "high", "low", "close"]] *= 0.70         # a real -30% day
    _, rep = validate_prices(crash, "NYSE", NOW, "SPY")
    assert rep.fatal == [] and any(f["check"] == "outlier_return" for f in rep.flags)


def test_minor_ohlc_inconsistency_is_clamped_old_ones_only():
    df = _frame(days=pd.bdate_range("2025-11-03", "2026-11-13"))       # one bad row is < 0.1% of the rows
    i = df.index[(df["ticker"] == "BBB")][10]
    df.loc[i, "high"] = df.loc[i, "close"] * 0.999                       # high below close, mid-history
    out, rep = validate_prices(df, "NYSE", NOW, "SPY")
    assert rep.fatal == [] and rep.clamped == 1
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] * 0.9    # on a last bar: fatal
    _, rep = validate_prices(df, "NYSE", NOW, "SPY")
    assert any("inconsistent OHLC" in f for f in rep.fatal)


def test_unscheduled_market_closure_is_reported_once():
    df = _frame()
    gone = df["date"] == pd.Timestamp("2026-10-15", tz="UTC")           # every ticker missing the same day
    _, rep = validate_prices(df[~gone], "NYSE", NOW, "SPY")
    checks = [f["check"] for f in rep.flags]
    assert checks.count("calendar_extra_session") == 1 and "missing_sessions" not in checks
    one = df[~(gone & (df["ticker"] == "CCC"))]                          # only one ticker missing it
    _, rep = validate_prices(one, "NYSE", NOW, "SPY")
    assert any(f["check"] == "missing_sessions" and f["ticker"] == "CCC" for f in rep.flags)

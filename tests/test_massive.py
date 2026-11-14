import numpy as np
import pandas as pd
import pytest

from research.massive.universe import MinuteResolver, pit_universe, split_reused_tickers
from stocks_etl.labeling import triple_barrier_ohlc


def _tb(o, h, l, c, side=1, sigma=0.05, t_max=5, **kw):
    return triple_barrier_ohlc(np.array(o), np.array(h), np.array(l), np.array(c), np.array([0]),
                               np.array([side]), np.array([sigma]), 1.0, 1.0, np.array([t_max]), **kw).iloc[0]


def test_intraday_touch_exits_at_barrier_not_close():
    # entry 100, TP 105: day 1 high 106 but closes 101 -> close-based labeling would miss it
    r = _tb([100, 100], [100, 106], [100, 99], [100, 101])
    assert r.barrier == "tp" and r.ret == pytest.approx(0.05) and r.end_pos == 1


def test_gap_through_stop_fills_at_open():
    r = _tb([100, 90], [100, 91], [100, 88], [100, 89])
    assert r.barrier == "sl" and r.ret == pytest.approx(-0.10)  # worse than the -5% stop


def test_short_side_mirrors_long():
    r = _tb([100, 100], [100, 101], [100, 94], [100, 99], side=-1)
    assert r.barrier == "tp" and r.ret == pytest.approx(0.05)


def test_both_touched_defaults_to_stop_and_resolver_can_flip_it():
    args = ([100, 100], [100, 106], [100, 94], [100, 100])
    r = _tb(*args)
    assert r.barrier == "sl" and r.ambiguous
    r = _tb(*args, resolve=lambda p, tp, sl, s: "tp")
    assert r.barrier == "tp" and r.ret == pytest.approx(0.05)


def test_time_barrier_and_delisting():
    flat = ([100] * 4, [101] * 4, [99] * 4, [100, 100, 100, 102])
    assert _tb(*flat, t_max=3).barrier == "t"
    r = _tb(*flat, t_max=10, series_ended=True)   # stock vanished with the trade open
    assert r.barrier == "delist" and r.ret == pytest.approx(0.02)
    assert _tb(*flat, t_max=10).barrier == "open"  # still trading: not labeled yet


def test_split_reused_tickers_separates_long_gaps():
    d = pd.bdate_range("2020-01-01", periods=40, tz="UTC")
    a = pd.DataFrame({"date": d[:5].append(d[30:]), "ticker": "ABC"})
    other = pd.DataFrame({"date": d, "ticker": "SPY"})
    out = split_reused_tickers(pd.concat([a, other], ignore_index=True))
    assert set(out["ticker"]) == {"ABC", "ABC~1", "SPY"}


def test_pit_universe_uses_only_past_and_keeps_delisted():
    d = pd.bdate_range("2020-01-01", periods=420, tz="UTC")
    rows = []
    for t, vol, n in (("BIG", 1e6, 420), ("DEAD", 2e6, 380), ("SMALL", 1e3, 420), ("ETF", 9e9, 420)):
        rows += [{"date": x, "ticker": t, "close": 50.0, "volume": vol} for x in d[:n]]
    df = pd.DataFrame(rows)
    m = pit_universe(df, cs={"BIG", "DEAD", "SMALL"}, top_n=2, min_history=300)
    first = m["date"].min()
    assert first > d[299]                     # needs 300 sessions of history first
    assert "ETF" not in set(m["ticker"])       # only common stocks are ranked
    early = m[m["date"] == first]["ticker"]
    assert set(early) == {"BIG", "DEAD"}       # DEAD is in while it trades...
    next_month = (d[379].tz_localize(None).to_period("M") + 1).start_time.tz_localize("UTC")
    assert "DEAD" not in set(m[m["date"] >= next_month]["ticker"])  # ...and gone from the next rebalance
    assert all(m.groupby("date").size() <= 2)


def test_minute_resolver_reads_cache_and_picks_first_touch(tmp_path):
    day = "2024-03-01"
    t0 = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
    bars = pd.DataFrame({"t": [int((t0 + pd.Timedelta(minutes=i)).timestamp() * 1000) for i in range(3)],
                         "h": [101, 106, 102], "l": [99, 98, 94]})
    (tmp_path / "XYZ").mkdir()
    bars.to_csv(tmp_path / "XYZ" / f"{day}.csv.gz", index=False, compression="gzip")
    res = MinuteResolver(client=None, cache=tmp_path)
    assert res("XYZ~1", pd.Timestamp(day), 105, 95, 1) == "tp"   # high 106 at 09:31 before low 94 at 09:32
    assert res("XYZ", pd.Timestamp(day), 105, 95, -1) == "sl"    # short: 106 >= stop 105 first
    assert res("NOPE", pd.Timestamp(day), 105, 95, 1) is None and res.unresolved == 1


def test_build_events_intraday_end_to_end():
    from stocks_etl.features import make_features
    from stocks_etl.meta_model import build_events
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2018-01-01", periods=600, tz="UTC")
    rows = []
    for t in ("SPY", "AAA", "BBB"):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(d))))
        o = c * np.exp(rng.normal(0, 0.003, len(d)))
        rows.append(pd.DataFrame({"date": d, "ticker": t, "open": o, "close": c,
                                  "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                                  "volume": 1e6}))
    feats, _ = make_features(pd.concat(rows, ignore_index=True), {"hmm_restarts": 1})
    cfg = {"barrier_mode": "intraday"}
    ev_c, _ = build_events(feats, {})
    ev_i, _ = build_events(feats, cfg)
    assert len(ev_i) > 0 and "ambiguous" in ev_i
    # intraday checks can only make exits earlier, never later
    m = ev_c.merge(ev_i, on=["ticker", "start"], suffixes=("_c", "_i"))
    assert (m["end_i"] <= m["end_c"]).all()
    feats["eligible"] = feats["ticker"] != "BBB"
    ev_e, _ = build_events(feats, cfg)
    assert "BBB" not in set(ev_e["ticker"])


def test_replay_candidates_respect_point_in_time_universe():
    from research.replay.paper_replay import candidates
    days = pd.bdate_range("2026-01-05", periods=4, tz="UTC")
    f = pd.DataFrame({"date": days, "ticker": "ZZA", "hurst": 0.6, "zscore": 1.0, "half_life": 5.0,
                      "log_ret_20": 0.05, "sigma_garch": 0.01, "regime": 0, "close": 100.0,
                      "eligible": [True, False, True, False]})
    c = candidates(f, {"regime_source": "asset"}, pd.Timestamp("2026-01-01"))
    assert list(c["day"].dt.day) == [5, 7]


def test_news_features_only_use_articles_before_the_close():
    from research.massive.news import news_features
    feats = pd.DataFrame({"date": pd.to_datetime(["2026-01-06", "2026-01-12"], utc=True), "ticker": "ZZA"})
    arts = pd.DataFrame({"ticker": "ZZA", "sent": [0.8, -0.4, 0.9],
                         "published_utc": pd.to_datetime(["2026-01-05T15:00Z",    # counted on the 6th
                                                          "2026-01-06T21:30Z",    # after the 6th's close
                                                          "2026-01-06T19:00Z"], utc=True)})
    out = news_features(arts, feats)
    assert out["news_count_5d"].tolist() == [2.0, 0.0]          # the 12th: all three are more than 5 days old
    assert out["news_sent_5d"].iloc[0] == pytest.approx((0.8 + 0.9) / 2)

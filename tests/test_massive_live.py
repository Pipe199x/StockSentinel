import pandas as pd
import pytest

from stocks_etl.stocks_etl import load_prices, massive_frame

# 2026-01-05 and 2026-01-06 sessions, as Massive stamps them (midnight New York, in ms)
T0 = int(pd.Timestamp("2026-01-05", tz="America/New_York").timestamp() * 1000)
T1 = int(pd.Timestamp("2026-01-06", tz="America/New_York").timestamp() * 1000)


def _rows(c0=100.0, c1=101.0):
    return [{"t": T0, "o": 99.0, "h": 101.0, "l": 98.0, "c": c0, "v": 1e6},
            {"t": T1, "o": 100.5, "h": 102.0, "l": 100.0, "c": c1, "v": 2e6}]


def test_massive_frame_matches_pipeline_schema_and_dates():
    df = massive_frame(_rows(), "AAA")
    assert list(df.columns) == ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
                                "dividends", "stock_splits"]
    # session day at 00:00 UTC, the convention the whole pipeline uses
    assert [str(d) for d in df["date"]] == ["2026-01-05 00:00:00+00:00", "2026-01-06 00:00:00+00:00"]
    assert df["close"].tolist() == [100.0, 101.0]


def test_massive_news_maps_to_raw_schema_and_flags_deal_spam():
    """Raw rows keep what the historical backfill kept (Massive id, every tag, spam not
    dropped) so the live sentiment table is built like the one the model was trained on."""
    from news_etl.news_client import massive_rows
    arts = [{"id": "abc123", "published_utc": "2026-01-05T14:00:00Z", "title": "Microsoft beats estimates",
             "description": "Cloud revenue up", "article_url": "https://x/1", "image_url": None,
             "publisher": {"name": "Reuters"}, "tickers": ["MSFT", "ORCL"]},
            {"id": "def456", "published_utc": "2026-01-05T15:00:00Z", "title": "Best Black Friday deals on Surface",
             "description": "", "article_url": "https://x/2", "publisher": {"name": "Blog"}, "tickers": ["MSFT"]}]
    rows = massive_rows(arts, "MSFT", {"MSFT", "AMZN"})
    assert len(rows) == 2                                   # spam is flagged, not dropped
    r, spam = rows
    assert r["raw_tickers"] == ["MSFT"] and r["source"] == "Reuters" and r["url"] == "https://x/1"
    assert r["id"] == "abc123" and r["all_tickers"] == ["MSFT", "ORCL"] and r["is_spam"] is False
    assert spam["is_spam"] is True
    assert set(r) == {"id", "published_at", "source", "title", "summary", "url", "image_url", "raw_tickers",
                      "all_tickers", "is_spam"}


class _FakeClient:
    def __init__(self, bars):
        self.bars, self.calls = bars, []

    def daily_aggs(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        return self.bars.get(ticker, [])


def test_load_prices_is_massive_only_and_keeps_nyse_sessions():
    sat = int(pd.Timestamp("2026-01-10", tz="America/New_York").timestamp() * 1000)   # a Saturday
    bars = {"AAA": _rows() + [{"t": sat, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0}], "BBB": _rows()}
    fake = _FakeClient(bars)
    cfg = {"start": "2016-11-14", "end": "2026-01-12", "massive_history_years": 10, "calendar": "NYSE"}
    df = load_prices(cfg, ["AAA", "BBB"], client=fake)
    assert df.groupby("ticker").size().to_dict() == {"AAA": 2, "BBB": 2}     # the Saturday bar is dropped
    assert list(df.columns)[:6] == ["date", "ticker", "open", "high", "low", "close"]
    # never earlier than the plan's rolling window, nor than cfg start
    assert all(start >= "2016-11-14" for _, start, _ in fake.calls)


def test_load_prices_fails_loudly_on_a_ticker_without_bars():
    with pytest.raises(SystemExit):
        load_prices({"end": "2026-01-12", "calendar": "NYSE"}, ["AAA", "ZZZ"], client=_FakeClient({"AAA": _rows()}))

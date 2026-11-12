import numpy as np
import pandas as pd

from etl_common.news_features import news_features, signed_sentiment, load_sentiment


def test_signed_sentiment():
    assert signed_sentiment("positive", 0.9) == 0.9
    assert signed_sentiment("negative", 0.8) == -0.8
    assert signed_sentiment("neutral", 0.7) == 0.0


def _feats(dates, ticker="AAA"):
    return pd.DataFrame({"date": pd.to_datetime(dates, utc=True), "ticker": ticker})


def test_news_features_are_point_in_time_and_windowed():
    # two signal days; articles before each day's 20:00 UTC close count, same-day-after-close does not
    scored = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "published_utc": pd.to_datetime(
            ["2026-01-05T10:00:00Z",   # before 2026-01-06 close, within 5d
             "2026-01-06T22:00:00Z",   # AFTER the 2026-01-06 close -> must not count that day
             "2025-12-20T10:00:00Z"],  # older than 5 days -> never counts
            utc=True),
        "sent": [0.8, -0.4, 0.9],
    })
    out = news_features(_feats(["2026-01-06"]), scored)
    assert out.loc[0, "news_count_5d"] == 1.0            # only the 01-05 article
    assert abs(out.loc[0, "news_sent_5d"] - 0.8) < 1e-9  # mean of just that article


def test_news_features_zero_without_news():
    out = news_features(_feats(["2026-01-06"], ticker="ZZZ"), pd.DataFrame(columns=["ticker", "published_utc", "sent"]))
    assert out["news_sent_5d"].tolist() == [0.0]
    assert out["news_count_5d"].tolist() == [0.0]


def test_load_sentiment_missing_file_is_empty(tmp_path):
    df = load_sentiment(tmp_path / "nope.csv")
    assert df.empty and list(df.columns) == ["id", "ticker", "published_utc", "sent"]

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


# --- live append built like the historical backfill (research/massive/news.py) -------------
from etl_common.news_features import _stable_id, append_sentiment  # noqa: E402

MASSIVE_ID = "tW4z_AjW6rXRCkv0HciadummxIUC1XZDp6uUq5TXfpQ"   # 43-char Massive article id


def _item(aid=MASSIVE_ID, url="https://x/1", tags=("MSFT",), pub="2026-11-12T15:00:00Z", label="positive"):
    return {"id": aid, "url": url, "published_at": pub, "all_tickers": list(tags), "raw_tickers": list(tags),
            "sentiment": {"label": label, "score": 0.9}}


def test_append_uses_the_massive_id_and_dedupes(tmp_path):
    path = tmp_path / "s.csv"
    assert append_sentiment([_item()], path, {"MSFT"}) == 1
    assert append_sentiment([_item()], path, {"MSFT"}) == 0           # same (id, ticker): not appended again
    df = load_sentiment(path)
    assert list(df["id"]) == [MASSIVE_ID]


def test_round_up_rule_uses_all_massive_tags(tmp_path):
    # 4 Massive tags (only one in the universe) = market round-up, like the backfill
    assert append_sentiment([_item(tags=("MSFT", "ORCL", "IBM", "SAP"))], tmp_path / "s.csv", {"MSFT"}) == 0
    assert append_sentiment([_item(tags=("MSFT", "ORCL"))], tmp_path / "t.csv", {"MSFT"}) == 1


def test_legacy_url_hash_rows_are_rekeyed_or_dropped(tmp_path):
    path = tmp_path / "s.csv"
    legacy = _stable_id({"url": "https://x/1"})
    pd.DataFrame({"id": [legacy, "c" * 16, MASSIVE_ID[::-1]], "ticker": ["MSFT", "AAPL", "AAPL"],
                  "published_utc": ["2026-11-12T15:00:00Z", "2026-11-12T16:00:00Z", "2026-11-12T16:00:00Z"],
                  "sent": [0.9, 0.5, 0.5]}).to_csv(path, index=False)
    # the AAPL legacy row duplicates a Massive-id row (same ticker and time): dropped on read
    assert sorted(load_sentiment(path)["id"]) == sorted([legacy, MASSIVE_ID[::-1]])
    # the MSFT article comes back with its Massive id: the legacy row is re-keyed, not duplicated
    assert append_sentiment([_item()], path, {"MSFT", "AAPL"}) == 0
    df = load_sentiment(path)
    assert len(df) == 2 and set(df["id"]) == {MASSIVE_ID, MASSIVE_ID[::-1]}

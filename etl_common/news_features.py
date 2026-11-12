"""Point-in-time news-sentiment features for the meta-model.

Shared by the research re-test (research/massive/news.py, scenario M3) and the live
pipeline (stocks_etl). Both compute the same two features from a table of signed
FinBERT sentiment, so what the live model uses is exactly what M3 validated:

  news_sent_5d   mean signed sentiment of articles published in the 5 days before the
                 signal day's close (20:00 UTC, conservative for EST and EDT), 0 without news
  news_count_5d  number of such articles

Signed sentiment = FinBERT P(label) with a sign: positive +p, negative -p, neutral 0.
An article tagged with more than MAX_TICKERS tickers is a market round-up, not news about
the stock, and is dropped upstream.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

WINDOW_DAYS = 5
CLOSE_UTC_HOUR = 20
MAX_TICKERS = 3
NEWS_FEATURES: List[str] = ["news_sent_5d", "news_count_5d"]

SENTIMENT_COLUMNS = ["id", "ticker", "published_utc", "sent"]


def signed_sentiment(label: str, score: float) -> float:
    """FinBERT label + confidence -> a single signed number in [-1, 1]."""
    label = (label or "").lower()
    if label == "positive":
        return float(score)
    if label == "negative":
        return -float(score)
    return 0.0


def load_sentiment(path: str | Path) -> pd.DataFrame:
    """Read the sentiment table; empty frame (right columns) if it is missing."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)
    df = pd.read_csv(p, dtype={"id": str})
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
    df["sent"] = df["sent"].astype(float)
    return df


def _stable_id(item: dict) -> str:
    import hashlib
    key = item.get("url") or f"{item.get('title', '')}|{item.get('published_at', '')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def append_sentiment(items, path: str | Path, universe) -> int:
    """Append signed-sentiment rows for FinBERT-enriched news items to the table.

    Each item needs 'sentiment' ({label, score}), 'published_at', and provider tags in
    'raw_tickers' (falls back to 'tickers'). One row per (article, universe ticker), deduped
    by (id, ticker) against what is already on disk. Keeps the meta-model's news current
    without re-downloading history. Best-effort: returns how many rows were added.
    """
    universe = set(universe or [])
    existing = load_sentiment(path)
    seen = set(zip(existing["id"], existing["ticker"])) if len(existing) else set()
    rows = []
    for it in items:
        s = it.get("sentiment") or {}
        if not s:
            continue
        sent = signed_sentiment(s.get("label", ""), s.get("score", 0.0))
        pub = it.get("published_at") or it.get("published_utc")
        tags = it.get("raw_tickers") or it.get("tickers") or []
        if len(tags) > MAX_TICKERS or not pub:
            continue
        aid = _stable_id(it)
        for t in tags:
            if t in universe and (aid, t) not in seen:
                rows.append((aid, t, pub, sent))
                seen.add((aid, t))
    if not rows:
        return 0
    add = pd.DataFrame(rows, columns=SENTIMENT_COLUMNS)
    add["published_utc"] = pd.to_datetime(add["published_utc"], utc=True)
    out = add if existing.empty else pd.concat([existing, add], ignore_index=True)
    out = out.sort_values("published_utc")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(rows)


def news_features(feats: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Add news_sent_5d and news_count_5d to feats using only articles known before each close.

    feats needs 'date' (tz-aware or naive UTC) and 'ticker'. scored needs 'ticker',
    'published_utc' (tz-aware UTC) and 'sent'. The window is strictly before the close,
    so there is no look-ahead.
    """
    out = feats.copy()
    out["news_sent_5d"] = 0.0
    out["news_count_5d"] = 0.0
    if scored is None or scored.empty:
        return out
    day = pd.to_datetime(out["date"], utc=True).dt.normalize()
    close = day + pd.Timedelta(hours=CLOSE_UTC_HOUR)
    for t, g in scored.groupby("ticker"):
        m = (out["ticker"] == t).to_numpy()
        if not m.any():
            continue
        g = g.sort_values("published_utc")
        s = g["sent"].fillna(0.0).to_numpy()
        tsv = g["published_utc"].to_numpy()
        csum = np.concatenate([[0.0], np.cumsum(s)])
        hi = np.searchsorted(tsv, close[m].to_numpy(), side="left")                    # strictly before the close
        lo = np.searchsorted(tsv, (close[m] - pd.Timedelta(days=WINDOW_DAYS)).to_numpy(), side="left")
        n = hi - lo
        out.loc[m, "news_count_5d"] = n.astype(float)
        out.loc[m, "news_sent_5d"] = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), 0.0)
    return out

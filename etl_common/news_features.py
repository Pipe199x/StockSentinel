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


def _read_sentiment(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)
    df = pd.read_csv(p, dtype={"id": str})
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
    df["sent"] = df["sent"].astype(float)
    return df


def _is_legacy_id(ids: pd.Series) -> pd.Series:
    """Rows written by live runs before 2026-11-14 used sha1(url)[:16] instead of Massive's id."""
    return ids.astype(str).str.fullmatch(r"[0-9a-f]{16}")


def drop_legacy_twins(df: pd.DataFrame) -> pd.DataFrame:
    """Drop legacy url-hash rows that duplicate a Massive-id row (same ticker and timestamp):
    the same article counted twice would inflate news_count_5d."""
    if df.empty:
        return df
    legacy = _is_legacy_id(df["id"])
    if not legacy.any():
        return df
    massive = set(zip(df.loc[~legacy, "ticker"], df.loc[~legacy, "published_utc"]))
    twin = np.fromiter(((t, p) in massive for t, p in zip(df["ticker"], df["published_utc"])), bool, len(df))
    return df[~(legacy.to_numpy() & twin)].reset_index(drop=True)


def load_sentiment(path: str | Path) -> pd.DataFrame:
    """Read the sentiment table (duplicated legacy rows removed); empty frame if it is missing."""
    return drop_legacy_twins(_read_sentiment(path))


def _stable_id(item: dict) -> str:
    import hashlib
    key = item.get("url") or f"{item.get('title', '')}|{item.get('published_at', '')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def append_sentiment(items, path: str | Path, universe) -> int:
    """Append signed-sentiment rows for FinBERT-enriched news items to the table.

    Built exactly like the historical backfill the meta-model was trained on
    (research/massive/news.py): one row per (Massive article id, tagged universe ticker);
    an article with more than MAX_TICKERS Massive tags (all of them, not only universe
    ones) is a market round-up and is skipped; spam is not filtered. Items need
    'sentiment' ({label, score}), 'published_at', 'id' and 'all_tickers' (older items
    without them fall back to the url hash and 'raw_tickers'/'tickers').

    Self-healing: a row first stored under the legacy url hash is re-keyed to the Massive
    id when that article is fetched again, and legacy rows duplicating a Massive-id row
    are dropped. Best-effort: returns how many new rows were added.
    """
    universe = set(universe or [])
    raw = _read_sentiment(path)
    existing = drop_legacy_twins(raw)
    healed = len(existing) < len(raw)
    seen = set(zip(existing["id"], existing["ticker"])) if len(existing) else set()
    rows, rekey = [], {}
    for it in items:
        s = it.get("sentiment") or {}
        if not s:
            continue
        sent = signed_sentiment(s.get("label", ""), s.get("score", 0.0))
        pub = it.get("published_at") or it.get("published_utc")
        tags = it.get("all_tickers") or it.get("raw_tickers") or it.get("tickers") or []
        if len(tags) > MAX_TICKERS or not pub:
            continue
        legacy = _stable_id(it)
        aid = it.get("id") or legacy
        for t in tags:
            if t not in universe or (aid, t) in seen:
                continue
            if aid != legacy and (legacy, t) in seen:   # stored before under its url hash: re-key it
                rekey[(legacy, t)] = aid
            else:
                rows.append((aid, t, pub, sent))
            seen.add((aid, t))
    if not rows and not rekey and not healed:
        return 0
    out = existing.copy()
    if rekey:
        out["id"] = [rekey.get((i, t), i) for i, t in zip(out["id"], out["ticker"])]
    if rows:
        add = pd.DataFrame(rows, columns=SENTIMENT_COLUMNS)
        add["published_utc"] = pd.to_datetime(add["published_utc"], utc=True)
        out = add if out.empty else pd.concat([out, add], ignore_index=True)
    out = out.sort_values("published_utc", kind="stable")
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

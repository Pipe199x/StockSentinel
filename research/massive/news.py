"""Massive news history + FinBERT scores for scenario M3 (does news add to V4?).

    python -m research.massive.news            # download (resumable) + score (cached)

Rules fixed before looking at results:
  - articles tagged with more than MAX_TICKERS tickers are dropped (market round-ups, not news
    about the stock);
  - an article counts for a signal day only if it was published before that day's close
    (20:00 UTC, conservative for both EST and EDT) and within the previous 5 calendar days;
  - sentiment = FinBERT P(label) signed: positive +p, negative -p, neutral 0.
Features per (ticker, day): news_sent_5d (mean signed sentiment, 0 without news) and
news_count_5d (number of articles).
Raw news stays in data/massive/news/ (gitignored, licensed for personal use).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.massive.client import MassiveClient, load_api_key

NEWS = Path("data/massive/news")
MAX_TICKERS = 3
WINDOW_DAYS = 5
CLOSE_UTC_HOUR = 20


def download(client: MassiveClient, tickers, start: str) -> None:
    NEWS.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        path = NEWS / f"{t}.json"
        if path.exists():
            continue
        rows = []
        for a in client.paginate("/v2/reference/news", {"ticker": t, "published_utc.gte": start,
                                                          "order": "asc", "sort": "published_utc", "limit": 1000}):
            rows.append({"id": a.get("id"), "published_utc": a.get("published_utc"), "title": a.get("title"),
                         "description": a.get("description"), "tickers": a.get("tickers") or [],
                         "publisher": (a.get("publisher") or {}).get("name")})
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows), encoding="utf-8")
        tmp.replace(path)
        print(f"[news] {t}: {len(rows)} articles")


def load_articles(tickers) -> pd.DataFrame:
    """One row per (article, ticker) for the requested tickers, round-ups removed, deduplicated."""
    keep = set(tickers)
    rows = []
    for t in tickers:
        path = NEWS / f"{t}.json"
        if not path.exists():
            continue
        for a in json.loads(path.read_text(encoding="utf-8")):
            if len(a["tickers"]) > MAX_TICKERS or t not in a["tickers"]:
                continue
            rows.append({**a, "ticker": t})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["ticker"].isin(keep)].drop_duplicates(["id", "ticker"])
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
    return df.reset_index(drop=True)


def score(df: pd.DataFrame, cache: Path = NEWS / "finbert_scores.csv") -> pd.DataFrame:
    """Signed FinBERT sentiment per article id, cached (FinBERT on CPU is the slow part)."""
    from news_etl.finbert_sentiment import FinBertSentiment
    done = pd.read_csv(cache, dtype={"id": str}) if cache.exists() else pd.DataFrame(columns=["id", "sent"])
    todo = df.drop_duplicates("id")
    todo = todo[~todo["id"].isin(set(done["id"]))]
    print(f"[finbert] {len(todo)} articles to score ({len(done)} cached)")
    fb = FinBertSentiment()
    sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    for i in range(0, len(todo), 2000):
        chunk = todo.iloc[i:i + 2000]
        items = [{"title": r.title or "", "summary": r.description or ""} for r in chunk.itertuples()]
        fb.enrich(items)
        out = pd.DataFrame({"id": chunk["id"].to_numpy(),
                            "sent": [sign.get((it.get("sentiment") or {}).get("label"), 0.0) *
                                     float((it.get("sentiment") or {}).get("score", 0.0)) for it in items]})
        done = pd.concat([done, out], ignore_index=True)
        done.to_csv(cache, index=False)
        print(f"[finbert] {min(i + 2000, len(todo))}/{len(todo)}")
    return df.merge(done, on="id", how="left")


def news_features(scored: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Adds news_sent_5d and news_count_5d to feats using only articles known before each close."""
    out = feats.copy()
    out["news_sent_5d"] = 0.0
    out["news_count_5d"] = 0.0
    if scored.empty:
        return out
    day = pd.to_datetime(out["date"]).dt.tz_convert("UTC").dt.normalize()
    close = day + pd.Timedelta(hours=CLOSE_UTC_HOUR)
    for t, g in scored.groupby("ticker"):
        m = (out["ticker"] == t).to_numpy()
        if not m.any():
            continue
        g = g.sort_values("published_utc")
        s = g["sent"].fillna(0.0).to_numpy()
        tsv = g["published_utc"].to_numpy()
        csum = np.concatenate([[0.0], np.cumsum(s)])
        hi = np.searchsorted(tsv, close[m].to_numpy(), side="left")          # strictly before the close
        lo = np.searchsorted(tsv, (close[m] - pd.Timedelta(days=WINDOW_DAYS)).to_numpy(), side="left")
        n = hi - lo
        out.loc[m, "news_count_5d"] = n.astype(float)
        out.loc[m, "news_sent_5d"] = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), 0.0)
    return out


def main():
    import yaml
    from stocks_etl.stocks_etl import universe_symbols
    cfg = yaml.safe_load(open("stocks_etl/config.yaml", encoding="utf-8"))
    tickers = [t for t in universe_symbols(cfg) if t not in ("SPY", "QQQ", "GLD", "TLT")]
    download(MassiveClient(load_api_key()), tickers, "2016-11-14")
    arts = load_articles(tickers)
    print(f"[news] {len(arts)} (article, ticker) rows after dropping round-ups")
    score(arts)


if __name__ == "__main__":
    main()

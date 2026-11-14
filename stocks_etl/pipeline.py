"""One way to build the model's feature frame: prices -> features -> point-in-time news.

Shared by the daily ETL, the weekly deep job, the advisor package runner and the stress
test, so every consumer sees exactly the inputs the live meta-model uses (the deep job's
XGBoost baseline used to skip the news features).
"""
import os
from typing import Dict, Tuple

import pandas as pd

from etl_common.news_features import NEWS_FEATURES, load_sentiment, news_features
from stocks_etl.features import make_features


def wants_news(qcfg: dict) -> bool:
    return any(c in (qcfg.get("extra_features") or []) for c in NEWS_FEATURES)


def build_feature_frame(prices: pd.DataFrame, qcfg: dict, docs_dir: str,
                        verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """make_features + (if the config asks for them) news_sent_5d / news_count_5d from the
    published sentiment table. Without the table the news features stay at 0."""
    feats, meta = make_features(prices, qcfg)
    if wants_news(qcfg):
        scored = load_sentiment(os.path.join(docs_dir, "history", "news_sentiment.csv"))
        feats = news_features(feats, scored)
        if verbose:
            print(f"[SUMMARY] news sentiment rows={len(scored)} "
                  f"(events with news={int((feats['news_count_5d'] > 0).sum())})")
    return feats, meta

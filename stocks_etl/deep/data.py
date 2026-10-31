"""Tensores para los modelos deep: secuencias de features y noticias alineadas sin mirar al futuro."""
import json
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stocks_etl.features import MODEL_FEATURES

STATIC = ["side", "strategy"]


def build_sequences(feats: pd.DataFrame, keys: pd.DataFrame, lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Para cada (ticker, start) devuelve las `lookback` filas de features que terminan en start.

    Devuelve (X [n, lookback, F], válido [n]). Eventos sin historia suficiente quedan inválidos.
    """
    n, F = len(keys), len(MODEL_FEATURES)
    X = np.zeros((n, lookback, F), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    for tkr, g in feats.groupby("ticker", sort=False):
        g = g.sort_values("date")
        dates = g["date"].dt.tz_localize(None) if g["date"].dt.tz is not None else g["date"]
        pos = pd.Series(np.arange(len(g)), index=dates.to_numpy())
        mat = g[MODEL_FEATURES].to_numpy(dtype=np.float32)
        sel = np.flatnonzero(keys["ticker"].to_numpy() == tkr)
        for i in sel:
            p = pos.get(keys["start"].iloc[i])
            if p is None or p + 1 < lookback:
                continue
            X[i] = mat[p + 1 - lookback:p + 1]
            ok[i] = True
    return X, ok


def standardize(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Escala con media/desv. del train únicamente (la del test sería fuga de información)."""
    flat = X[train_idx].reshape(-1, X.shape[-1])
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_news(history_path: str, emb_path: str) -> Optional[pd.DataFrame]:
    """Artículos con embedding: columnas ticker, published (UTC naive), emb_idx. None si no hay datos."""
    from news_etl.finbert_sentiment import article_id

    try:
        with np.load(emb_path, allow_pickle=False) as z:
            ids, emb = list(z["ids"]), z["emb"].astype(np.float32)
    except FileNotFoundError:
        return None
    idx = {a: i for i, a in enumerate(ids)}
    rows = []
    with open(history_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            it = json.loads(line)
            j = idx.get(article_id(it))
            if j is None or not it.get("published_at"):
                continue
            ts = pd.Timestamp(it["published_at"]).tz_convert("UTC").tz_localize(None)
            for t in it.get("tickers") or []:
                rows.append({"ticker": t, "published": ts, "emb_idx": j})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("published")
    df.attrs["emb"] = emb
    return df


def news_tensor(keys: pd.DataFrame, news: pd.DataFrame, days: int = 5, max_items: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Embeddings de las noticias de los `days` días previos a cada evento.

    Corte conservador: solo artículos publicados ANTES del día del evento (nada del mismo
    día, que podría salir después del cierre). Devuelve (E [n, max_items, 768], mask [n, max_items]).
    """
    emb = news.attrs["emb"]
    n = len(keys)
    E = np.zeros((n, max_items, emb.shape[1]), dtype=np.float32)
    M = np.zeros((n, max_items), dtype=bool)
    by_tkr: Dict[str, pd.DataFrame] = {t: g for t, g in news.groupby("ticker")}
    for i, (tkr, start) in enumerate(zip(keys["ticker"], keys["start"])):
        g = by_tkr.get(tkr)
        if g is None:
            continue
        day0 = pd.Timestamp(start).normalize()
        w = g[(g["published"] < day0) & (g["published"] >= day0 - timedelta(days=days))].tail(max_items)
        k = len(w)
        if k:
            E[i, :k] = emb[w["emb_idx"].to_numpy()]
            M[i, :k] = True
    return E, M


def news_coverage(keys: pd.DataFrame, news: Optional[pd.DataFrame]) -> np.ndarray:
    """Eventos dentro del periodo y universo cubiertos por el histórico de noticias."""
    if news is None:
        return np.zeros(len(keys), dtype=bool)
    first = news["published"].min().normalize()
    return keys["ticker"].isin(set(news["ticker"])).to_numpy() & (keys["start"] > first).to_numpy()

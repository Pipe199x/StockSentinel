import numpy as np
import pandas as pd
import pytest

from stocks_etl.deep.data import build_sequences, news_tensor, standardize
from stocks_etl.features import MODEL_FEATURES


def _feats(n=40):
    dates = pd.bdate_range("2026-01-01", periods=n, tz="UTC")
    f = pd.DataFrame({"date": dates, "ticker": "AAA"})
    for i, c in enumerate(MODEL_FEATURES):
        f[c] = np.arange(n, dtype=float) + i
    return f


def test_sequences_end_at_event_and_never_include_the_future():
    f = _feats()
    keys = pd.DataFrame({"ticker": ["AAA", "AAA"], "start": [f["date"].iloc[25].tz_localize(None),
                                                             f["date"].iloc[3].tz_localize(None)]})
    X, ok = build_sequences(f, keys, lookback=10)
    assert ok.tolist() == [True, False]  # el segundo no tiene 10 sesiones de historia
    assert X[0, -1, 0] == 25 and X[0, 0, 0] == 16


def test_standardize_uses_train_statistics_only():
    X = np.zeros((4, 2, 1), dtype=np.float32)
    X[2:] = 100.0
    Z = standardize(X, np.array([0, 1]))
    assert np.all(Z[:2] == 0) and np.all(Z[2:] == 100)  # desv. 0 en train -> escala 1, media 0


def test_news_window_excludes_same_day_articles():
    news = pd.DataFrame({
        "ticker": ["AAA"] * 3,
        "published": pd.to_datetime(["2026-03-02 15:00", "2026-03-04 09:00", "2026-03-05 14:00"]),
        "emb_idx": [0, 1, 2],
    })
    news.attrs["emb"] = np.eye(3, 768, dtype=np.float32)
    keys = pd.DataFrame({"ticker": ["AAA"], "start": [pd.Timestamp("2026-03-05")]})
    E, M = news_tensor(keys, news, days=5, max_items=4)
    assert M[0].sum() == 2  # el artículo del 5 de marzo (mismo día) queda fuera
    assert E[0, 1, 1] == 1.0 and E[0, :, 2].sum() == 0


def test_models_forward_shapes():
    torch = pytest.importorskip("torch")
    from stocks_etl.deep.models import FusionMeta, LSTMMeta, fit_predict

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 10, 5)).astype(np.float32)
    S = rng.integers(-1, 2, size=(64, 2)).astype(np.float32)
    y = rng.integers(0, 2, size=64)
    E = rng.normal(size=(64, 3, 768)).astype(np.float32)
    M = np.zeros((64, 3), dtype=bool)
    M[::2, 0] = True  # la mitad sin noticias: el token nulo evita una atención vacía
    tr, te = np.arange(48), np.arange(48, 64)
    for cls, kw in [(LSTMMeta, {}), (FusionMeta, {"E": E, "M": M})]:
        p = fit_predict(cls, X, S, y, tr, te, epochs=1, hidden=8, **kw)
        assert p.shape == (16,) and np.all((p >= 0) & (p <= 1))

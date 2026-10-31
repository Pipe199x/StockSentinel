"""Features estacionarias y de régimen (sin indicadores clásicos rezagados).

Se descartan RSI, medias móviles y niveles de precio/volumen: no son estacionarios y,
como insiste Julián, un indicador rezagado solo mira el retrovisor. En su lugar:
retornos log, diferenciación fraccionaria (estacionaria pero con memoria), volatilidad
realizada y GARCH, Hurst, distancia OU al valor justo y régimen HMM filtrado.
"""
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from stocks_etl.regime import fit_garch, hmm_filtered_regimes, hurst_rolling, ou_features

# Columnas que ve el meta-modelo (además de 'side', que añade primary.py)
MODEL_FEATURES = [
    "log_ret_1", "log_ret_5", "log_ret_20",
    "rv_10", "rv_20", "sigma_garch",
    "ffd_close", "hurst", "half_life", "zscore",
    "vol_z", "beta_spy", "p_stress", "regime",
    "mkt_p_stress", "mkt_regime",
]


def ffd_weights(d: float, thres: float = 1e-3, max_width: int = 120) -> np.ndarray:
    """Pesos de la diferenciación fraccionaria de ventana fija (w_0 = 1, w_1 = -d, ...)."""
    w = [1.0]
    k = 1
    while len(w) < max_width:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w)


def ffd(x: pd.Series, d: float, thres: float = 1e-3, max_width: int = 120) -> pd.Series:
    """y_t = Σ_k w_k · x_{t-k}. Las primeras len(w)-1 posiciones quedan en NaN."""
    w = ffd_weights(d, thres, max_width)
    vals = x.to_numpy(dtype=float)
    y = np.convolve(vals, w)[: len(vals)]
    y[: len(w) - 1] = np.nan
    return pd.Series(y, index=x.index)


def min_ffd_d(x: pd.Series, p_value: float = 0.05) -> float:
    """El d más pequeño que vuelve estacionaria la serie (ADF), conservando máxima memoria.

    Se elige con la primera mitad de la historia para no mirar el futuro.
    """
    from statsmodels.tsa.stattools import adfuller

    head = x.iloc[: len(x) // 2]
    for d in np.round(np.arange(0.1, 1.01, 0.1), 2):
        y = ffd(head, d).dropna()
        if len(y) < 50:
            continue
        if adfuller(y, maxlag=1, regression="c", autolag=None)[1] < p_value:
            return float(d)
    return 1.0


def _ticker_features(dfg: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, Dict]:
    dfg = dfg[dfg["close"].notna()].copy()
    logp = np.log(dfg["close"])
    dfg["log_ret_1"] = logp.diff()
    dfg["log_ret_5"] = logp.diff(5)
    dfg["log_ret_20"] = logp.diff(20)
    dfg["rv_10"] = dfg["log_ret_1"].rolling(10, min_periods=5).std()
    dfg["rv_20"] = dfg["log_ret_1"].rolling(20, min_periods=10).std()

    d = min_ffd_d(logp)
    dfg["ffd_close"] = ffd(logp, d)

    dfg["hurst"] = hurst_rolling(logp, window=int(cfg.get("hurst_window", 100)))
    ou = ou_features(logp, span=int(cfg.get("ou_span", 20)), window=int(cfg.get("ou_window", 100)))
    dfg["zscore"] = ou["zscore"]
    # vida media infinita = no hay reversión; se recorta para que el modelo la entienda
    dfg["half_life"] = ou["half_life"].clip(upper=60)

    logv = np.log(dfg["volume"].where(dfg["volume"] > 0))
    dfg["vol_z"] = (logv - logv.rolling(60, min_periods=20).mean()) / logv.rolling(60, min_periods=20).std()

    sigma, garch = fit_garch(dfg["log_ret_1"])
    dfg["sigma_garch"] = sigma

    probs = hmm_filtered_regimes(dfg[["log_ret_1", "rv_20"]].to_numpy(), seed=int(cfg.get("seed", 42)),
                                 restarts=int(cfg.get("hmm_restarts", 1)))
    dfg["p_calm"], dfg["p_normal"], dfg["p_stress"] = probs[:, 0], probs[:, 1], probs[:, 2]
    dfg["regime"] = np.where(np.isnan(probs).any(axis=1), np.nan, np.nanargmax(np.nan_to_num(probs, nan=-1), axis=1))

    return dfg, {"ffd_d": d, "garch": garch}


def make_features(df: pd.DataFrame, cfg: dict = None) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Espera columnas date,ticker,close,volume. Devuelve (features, metadatos por ticker)."""
    cfg = cfg or {}
    if df.empty:
        return df, {}
    df = df.sort_values(["ticker", "date"])
    out, meta = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tkr, dfg in df.groupby("ticker", sort=False):
            if dfg["close"].notna().sum() < 300:
                print(f"[FEATURES] {tkr}: historia insuficiente, se omite")
                continue
            feats, m = _ticker_features(dfg, cfg)
            out.append(feats)
            meta[tkr] = m
    feats = pd.concat(out, ignore_index=True)

    # beta frente al mercado (SPY): cuánto del movimiento es sistemático
    bench = cfg.get("benchmark", "SPY")
    wide = feats.pivot(index="date", columns="ticker", values="log_ret_1")
    if bench in wide.columns:
        m = wide[bench]
        var_m = m.rolling(60, min_periods=30).var()
        beta = wide.apply(lambda c: c.rolling(60, min_periods=30).cov(m) / var_m)
        beta_long = beta.stack(future_stack=True).rename("beta_spy").reset_index()
        feats = feats.merge(beta_long, on=["date", "ticker"], how="left")
    else:
        feats["beta_spy"] = np.nan

    # régimen de MERCADO (HMM del benchmark): el estrés de un solo activo suele ser
    # solo su tercil de volatilidad alta; la crisis que apaga la operativa es la del mercado
    if bench in meta:
        mkt = feats.loc[feats["ticker"] == bench, ["date", "regime", "p_stress"]]
        mkt = mkt.rename(columns={"regime": "mkt_regime", "p_stress": "mkt_p_stress"})
        feats = feats.merge(mkt, on="date", how="left")
    else:
        feats["mkt_regime"] = feats["regime"]
        feats["mkt_p_stress"] = feats["p_stress"]
    return feats, meta

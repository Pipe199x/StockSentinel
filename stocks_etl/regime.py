"""Detección de régimen: Hurst, GARCH(1,1)-t, HMM y vida media Ornstein-Uhlenbeck.

Todo lo que entra al modelo está *filtrado* (usa solo el pasado). El HMM no usa
Viterbi ni forward-backward, que miran el futuro: se aplica el filtro forward a mano.
"""
import math
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

REGIME_NAMES = {0: "calm", 1: "normal", 2: "stress"}


def hurst_rolling(log_price: pd.Series, window: int = 100, lags=range(2, 21)) -> pd.Series:
    """Exponente de Hurst por varianza de diferencias: std(x_{t+L} - x_t) ∝ L^H.

    H > 0.5 persistente (tendencia), H < 0.5 antipersistente (reversión).
    """
    lags = np.array(list(lags), dtype=float)
    taus = np.column_stack([
        np.log((log_price - log_price.shift(int(L))).rolling(window, min_periods=window // 2).std())
        for L in lags
    ])
    x = np.log(lags)
    xc = x - x.mean()
    slope = ((taus - taus.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / (xc ** 2).sum()
    return pd.Series(slope, index=log_price.index)


def ou_features(log_price: pd.Series, span: int = 20, window: int = 100, z_window: int = 60) -> pd.DataFrame:
    """Distancia al valor justo (EWM del log precio), su z-score y vida media OU rodante.

    Se modela la distancia a la media, no el precio: Δs_t = a + b·s_{t-1}; vida media = -ln2/ln(1+b).
    """
    s = log_price - log_price.ewm(span=span, adjust=False).mean()
    s_lag = s.shift(1)
    ds = s - s_lag
    b = ds.rolling(window, min_periods=window // 2).cov(s_lag) / s_lag.rolling(window, min_periods=window // 2).var()
    valid = (b < 0) & (b > -1)
    half_life = pd.Series(np.where(valid, -math.log(2) / np.log1p(b.where(valid, -0.5)), np.inf), index=s.index)
    half_life[b.isna()] = np.nan
    z = s / s.rolling(z_window, min_periods=z_window // 2).std()
    return pd.DataFrame({"ou_spread": s, "half_life": half_life, "zscore": z})


def fit_garch(log_ret: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """GARCH(1,1) con innovaciones t-Student. Devuelve σ condicional diaria y parámetros.

    alpha + beta < 1 garantiza que la volatilidad revierte a su media; cerca de 1 los
    shocks tardan mucho en disiparse.
    """
    from arch import arch_model

    r = log_ret.dropna() * 100.0
    res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off", show_warning=False)
    sigma = (res.conditional_volatility / 100.0).reindex(log_ret.index)
    p = res.params
    fc = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]) / 100.0)
    params = {
        "mu": float(p["mu"]) / 100.0,
        "omega": float(p["omega"]),
        "alpha": float(p["alpha[1]"]),
        "beta": float(p["beta[1]"]),
        "nu": float(p["nu"]),
        "sigma_next": fc,
    }
    params["persistence"] = params["alpha"] + params["beta"]
    params["sigma_long_run"] = (
        math.sqrt(params["omega"] / (1 - params["persistence"])) / 100.0 if params["persistence"] < 1 else float("nan")
    )
    return sigma, params


def hmm_filtered_regimes(X: np.ndarray, n_states: int = 3, seed: int = 42, restarts: int = 1) -> np.ndarray:
    """Probabilidades filtradas P(estado_t | x_1..x_t) de un HMM gaussiano.

    Los estados se ordenan por la media de la columna 1 (volatilidad) para que las
    etiquetas no cambien entre reentrenamientos: 0 = calma, 1 = normal, 2 = estrés.
    EM cae en óptimos locales: con una sola inicialización un día más de datos puede
    duplicar los días de estrés. Se reinicia `restarts` veces y se queda el de mayor
    verosimilitud. Filas con NaN reciben NaN.
    """
    from hmmlearn.hmm import GaussianHMM

    X = np.asarray(X, dtype=float)
    ok = np.isfinite(X).all(axis=1)
    out = np.full((len(X), n_states), np.nan)
    Xo = X[ok]
    mu, sd = Xo.mean(axis=0), Xo.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Xo - mu) / sd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, best = None, -np.inf
        for k in range(max(1, restarts)):
            m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=seed + k)
            m.fit(Z)
            score = m.score(Z)
            if np.isfinite(score) and score > best:
                model, best = m, score
    order = np.argsort(model.means_[:, 1])
    log_b = np.column_stack([
        multivariate_normal.logpdf(Z, mean=model.means_[k], cov=model.covars_[k], allow_singular=True)
        for k in order
    ])
    log_A = np.log(model.transmat_[np.ix_(order, order)] + 1e-300)
    log_alpha = np.log(model.startprob_[order] + 1e-300) + log_b[0]
    log_alpha -= logsumexp(log_alpha)
    filt = np.empty_like(log_b)
    filt[0] = log_alpha
    for t in range(1, len(Z)):
        log_alpha = logsumexp(log_alpha[:, None] + log_A, axis=0) + log_b[t]
        log_alpha -= logsumexp(log_alpha)
        filt[t] = log_alpha
    out[ok] = np.exp(filt)
    return out

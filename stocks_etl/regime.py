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
    fc = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]) / 100.0)
    return sigma, garch_param_dict(res.params, fc)


def garch_param_dict(p, sigma_next: float) -> Dict[str, float]:
    """arch parameters (returns in %) -> the dict the pipeline publishes. With a GJR term
    (gamma, the leverage effect) persistence is alpha + beta + gamma/2."""
    gamma = float(p["gamma[1]"]) if "gamma[1]" in p.index else 0.0
    params = {
        "mu": float(p["mu"]) / 100.0,
        "omega": float(p["omega"]),
        "alpha": float(p["alpha[1]"]),
        "beta": float(p["beta[1]"]),
        "gamma": gamma,
        "nu": float(p["nu"]),
        "sigma_next": float(sigma_next),
    }
    params["persistence"] = params["alpha"] + params["beta"] + gamma / 2.0
    params["sigma_long_run"] = (
        math.sqrt(params["omega"] / (1 - params["persistence"])) / 100.0 if params["persistence"] < 1 else float("nan")
    )
    return params


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


# --- Point-in-time (sin look-ahead) -------------------------------------------
# Los parámetros de GARCH y del HMM también son información: si se ajustan con toda la
# historia, el σ y el régimen de 2010 "conocen" 2020. Aquí se reajustan una vez por año
# con datos ANTERIORES a esa fecha y se aplican hacia adelante hasta el siguiente ajuste.

def refit_boundaries(dates: pd.Index, min_train: int, every_years: int = 1) -> list:
    """Posiciones donde se reajusta: primera sesión de cada año (o cada `every_years`)
    con al menos `min_train` observaciones previas. Siempre ordenadas."""
    years = pd.DatetimeIndex(dates).year
    out, last_year = [], None
    for i in range(1, len(years)):
        if years[i] != years[i - 1] and i >= min_train:
            if last_year is None or years[i] - last_year >= every_years:
                out.append(i)
                last_year = years[i]
    return out


def fit_garch_pit(log_ret: pd.Series, dates, min_train: int = 500, every_years: int = 1):
    """σ condicional y ν (grados de libertad) sin mirar el futuro.

    Para cada segmento [b_k, b_{k+1}) los parámetros se estiman con los datos < b_k y el
    filtro GARCH (que solo usa el pasado) corre con esos parámetros fijos. Antes del primer
    ajuste todo es NaN: no se opera en el periodo de entrenamiento inicial.
    Devuelve (sigma, nu, params_hoy). params_hoy son los parámetros del ÚLTIMO segmento
    point-in-time (los que dimensionan la operación de hoy) con su pronóstico a un día
    (sigma_next): el riesgo publicado es el mismo modelo que fija tamaño y barreras.
    """
    from arch import arch_model

    sigma = pd.Series(np.nan, index=log_ret.index)
    nu = pd.Series(np.nan, index=log_ret.index)
    r_all = log_ret * 100.0
    bounds = refit_boundaries(pd.DatetimeIndex(dates), min_train, every_years)
    today = None
    for k, b in enumerate(bounds):
        end = bounds[k + 1] if k + 1 < len(bounds) else len(log_ret)
        train = r_all.iloc[:b].dropna()
        if len(train) < min_train // 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off",
                                                                                         show_warning=False)
            seg = r_all.iloc[:end].dropna()
            fixed = arch_model(seg, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fix(res.params)
            if end == len(log_ret):  # last segment: one-step forecast with the point-in-time parameters
                fc = float(np.sqrt(fixed.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]) / 100.0)
                today = garch_param_dict(res.params, fc)
        cv = (fixed.conditional_volatility / 100.0).reindex(log_ret.index)
        sl = slice(b, end)
        sigma.iloc[sl] = cv.iloc[sl].to_numpy()
        nu.iloc[sl] = float(res.params["nu"])
    if today is None:  # history too short for a point-in-time fit: fall back to the full sample
        _, today = fit_garch(log_ret)
    return sigma, nu, today


def _hmm_fit(Z: np.ndarray, n_states: int, seed: int, restarts: int):
    from hmmlearn.hmm import GaussianHMM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, best = None, -np.inf
        for k in range(max(1, restarts)):
            m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=seed + k)
            m.fit(Z)
            score = m.score(Z)
            if np.isfinite(score) and score > best:
                model, best = m, score
    return model


def _hmm_forward(model, Z: np.ndarray) -> np.ndarray:
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
    return np.exp(filt)


def hmm_filtered_regimes_pit(X: np.ndarray, dates: pd.Index, min_train: int = 500, n_states: int = 3,
                             seed: int = 42, restarts: int = 3, every_years: int = 1) -> np.ndarray:
    """Como hmm_filtered_regimes, pero cada año el HMM (y la estandarización) se ajusta solo
    con filas anteriores al segmento; el filtro forward corre con ese modelo fijo."""
    X = np.asarray(X, dtype=float)
    out = np.full((len(X), n_states), np.nan)
    ok = np.isfinite(X).all(axis=1)
    bounds = refit_boundaries(pd.DatetimeIndex(dates), min_train, every_years)
    for k, b in enumerate(bounds):
        end = bounds[k + 1] if k + 1 < len(bounds) else len(X)
        tr = ok.copy()
        tr[b:] = False
        if tr.sum() < min_train // 2:
            continue
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd[sd == 0] = 1.0
        model = _hmm_fit((X[tr] - mu) / sd, n_states, seed, restarts)
        if model is None:
            continue
        rows = np.flatnonzero(ok[:end])
        probs = _hmm_forward(model, (X[rows] - mu) / sd)
        seg = rows >= b
        out[rows[seg]] = probs[seg]
    return out

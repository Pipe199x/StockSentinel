"""Gestión de riesgo: Kelly fraccionado, VaR/ES de colas pesadas, Monte Carlo t y rachas.

Principio de Julián: el trabajo cuant no es acertar cada vela sino que equivocarse
salga barato. Ninguna fórmula aquí supone retornos normales.
"""
import math
from typing import Dict

import numpy as np
from scipy.stats import norm, t as student_t


def kelly_fraction(p: float, payoff: float, frac: float = 0.5, cap: float = 1.0) -> float:
    """f* = p - (1-p)/b, recortado a [0, cap] y multiplicado por la fracción (≤ ½ Kelly)."""
    if p is None or not np.isfinite(p) or payoff <= 0:
        return 0.0
    f = p - (1.0 - p) / payoff
    return float(min(max(f, 0.0), cap) * frac)


def vol_target_scale(sigma_daily: float, target_annual: float = 0.20) -> float:
    """Reduce la exposición cuando la volatilidad sube (nunca apalanca por encima de 1)."""
    if not sigma_daily or not np.isfinite(sigma_daily):
        return 0.0
    return float(min(1.0, target_annual / (sigma_daily * math.sqrt(252))))


def var_es_t(sigma: float, nu: float, alpha: float, mu: float = 0.0) -> Dict[str, float]:
    """VaR y Expected Shortfall de una t-Student estandarizada (varianza = sigma²).

    Devuelve pérdidas como números positivos (fracción del capital).
    """
    nu = max(float(nu), 2.05)
    scale = sigma * math.sqrt((nu - 2) / nu)
    q = student_t.ppf(alpha, nu)
    var = -(mu + scale * q)
    es_std = student_t.pdf(q, nu) * (nu + q ** 2) / ((nu - 1) * alpha)
    es = -mu + scale * es_std
    return {"var": float(var), "es": float(es)}


def cornish_fisher_var(sigma: float, skew: float, exkurt: float, alpha: float, mu: float = 0.0) -> float:
    """VaR modificado que corrige el cuantil normal por asimetría y curtosis."""
    z = norm.ppf(alpha)
    zcf = (z + (z ** 2 - 1) * skew / 6 + (z ** 3 - 3 * z) * exkurt / 24
           - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)
    return float(-(mu + sigma * zcf))


def mc_t_paths(mu: float, sigma: float, nu: float, horizon: int, n_paths: int = 10000, seed: int = 42) -> np.ndarray:
    """Retornos acumulados a `horizon` días con innovaciones t-Student (colas pesadas)."""
    rng = np.random.default_rng(seed)
    nu = max(float(nu), 2.05)
    shocks = rng.standard_t(nu, size=(n_paths, horizon)) * math.sqrt((nu - 2) / nu)
    log_r = (mu - 0.5 * sigma ** 2) + sigma * shocks
    return np.expm1(log_r.sum(axis=1))


def expected_max_losing_streak(p_win: float, n_trades: int) -> float:
    """Racha perdedora típica más larga en n trades: ln(n·p) / -ln(q). No es un techo."""
    q = 1.0 - p_win
    if n_trades <= 0 or not 0 < q < 1 or n_trades * p_win <= 1:
        return float("nan")
    return float(math.log(n_trades * p_win) / -math.log(q))


def streak_drawdown(streak: float, risk_per_trade: float) -> float:
    """Pérdida acumulada tras `streak` pérdidas seguidas arriesgando una fracción fija."""
    if not np.isfinite(streak):
        return float("nan")
    return float(1.0 - (1.0 - risk_per_trade) ** streak)


def max_losing_streak(wins) -> int:
    best = cur = 0
    for w in wins:
        cur = 0 if w else cur + 1
        best = max(best, cur)
    return best

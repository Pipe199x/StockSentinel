"""Regime and volatility diagnostics the advisor teaches (informational, no trade changes).

  - HMM transition matrix: expected regime duration 1/(1 - p_ii), survival p_ii^d, current
    regime age and the chance of entering stress within 5 / 10 sessions (9 Nov video: with
    50-day regimes, lasting 100+ days has ~13% probability).
  - Training mix: a model trained in one regime and deployed in another can behave differently
    (8 Sep video), so warn when today's regime was rare among the out-of-sample trades.
  - GARCH: alpha + beta (+ gamma/2) < 1, long-run volatility, shock half-life
    ln(0.5) / ln(persistence) and a critical level at 1.5x the long-run volatility (14 Oct / 8 Nov).
  - Post-shock window: after a benchmark day <= -4%, the next 10 sessions carry up to 4x more
    large moves (8 Nov video).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REGIMES = {0: "calm", 1: "normal", 2: "stress"}


def transition_summary(transmat, regimes: pd.Series, p_now: Optional[np.ndarray] = None) -> Dict:
    """Duration and switch odds of the current market regime from the latest HMM segment."""
    A = np.asarray(transmat, float)
    reg = regimes.dropna().astype(int)
    if A.shape != (3, 3) or reg.empty:
        return {}
    cur = int(reg.iloc[-1])
    run = (reg != reg.shift()).cumsum()
    age = int((run == run.iloc[-1]).sum())
    stay = np.diag(A)
    exp_dur = {REGIMES[i]: (float(1 / (1 - stay[i])) if stay[i] < 1 else None) for i in range(3)}
    start = np.eye(3)[cur] if p_now is None else np.asarray(p_now, float)
    p5 = float(start @ np.linalg.matrix_power(_stress_absorbing(A), 5)[:, 2])
    p10 = float(start @ np.linalg.matrix_power(_stress_absorbing(A), 10)[:, 2])
    return {
        "current": REGIMES[cur], "age_sessions": age,
        "expected_duration_sessions": exp_dur,
        "p_stay_next_session": float(stay[cur]),
        "p_still_in_regime": {str(d): float(stay[cur] ** d) for d in (10, 20, 50, 100)},
        "p_stress_within_5": p5, "p_stress_within_10": p10,
        "transition_matrix": {REGIMES[i]: {REGIMES[j]: float(A[i, j]) for j in range(3)} for i in range(3)},
    }


def _stress_absorbing(A: np.ndarray) -> np.ndarray:
    B = np.array(A, dtype=float, copy=True)
    B[2] = 0.0
    B[2, 2] = 1.0
    return B


def training_mix(oof_trades: pd.DataFrame, current_regime: Optional[str], min_share: float = 0.15,
                 min_trades: int = 30) -> Dict:
    """Share of out-of-sample trades per market regime, and whether today's regime was rare."""
    if oof_trades is None or oof_trades.empty or "regime" not in oof_trades:
        return {}
    counts = oof_trades["regime"].dropna().astype(int).map(REGIMES).value_counts()
    total = int(counts.sum())
    share = {k: float(v / total) for k, v in counts.items()} if total else {}
    n_cur = int(counts.get(current_regime, 0)) if current_regime else 0
    rare = bool(current_regime) and current_regime != "stress" and (
        share.get(current_regime, 0.0) < min_share or n_cur < min_trades)
    return {"trades_by_regime": {k: int(v) for k, v in counts.items()}, "share": share,
            "current_regime": current_regime, "current_regime_rare_in_training": rare}


IDENTIFIED_PERSISTENCE = 0.995


def garch_diag(params: Dict, sigma_now: Optional[float] = None, alert_ratio: float = 1.5,
               sample_sigma: Optional[float] = None) -> Dict:
    """Persistence, long-run volatility, shock half-life and the critical-level alert (annualized).

    omega / (1 - persistence) is the long-run variance only while persistence is clearly below 1;
    near 1 (IGARCH-like fits, common on index ETFs) it explodes (SPY: 65% a year). Then the long-run
    reference is the sample volatility of the history, and the report says so."""
    if not params:
        return {}
    pers = float(params.get("persistence", np.nan))
    lr = params.get("sigma_long_run")
    s_now = float(sigma_now if sigma_now is not None else params.get("sigma_next", np.nan))
    ann = math.sqrt(252)
    half_life = float(math.log(0.5) / math.log(pers)) if 0 < pers < 1 else None
    if pers < IDENTIFIED_PERSISTENCE and lr is not None and np.isfinite(lr):
        lr_ann, source = float(lr * ann), "garch"
    elif sample_sigma is not None and np.isfinite(sample_sigma):
        lr_ann, source = float(sample_sigma * ann), "sample (GARCH persistence ~1: long-run variance not identified)"
    else:
        lr_ann, source = None, None
    now_ann = float(s_now * ann) if np.isfinite(s_now) else None
    ratio = float(now_ann / lr_ann) if now_ann and lr_ann else None
    return {"persistence": pers, "stationary": bool(pers < 1), "nu": params.get("nu"),
            "vol_now_annual": now_ann, "vol_long_run_annual": lr_ann, "long_run_source": source,
            "ratio_to_long_run": ratio, "critical_level_annual": float(lr_ann * alert_ratio) if lr_ann else None,
            "alert": bool(ratio is not None and ratio >= alert_ratio), "shock_half_life_sessions": half_life}


def post_shock(bench_ret: pd.Series, drop: float = -0.04, window: int = 10) -> Dict:
    """Active when the benchmark closed `drop` or worse within the last `window` sessions."""
    r = bench_ret.dropna()
    if r.empty:
        return {"active": False}
    simple = np.expm1(r.to_numpy(float))
    recent = simple[-window:]
    hits = np.flatnonzero(recent <= drop)
    out = {"active": bool(len(hits)), "threshold": drop, "window_sessions": window,
           "worst_recent_day": float(recent.min()) if len(recent) else None}
    if len(hits):
        out["sessions_since_shock"] = int(len(recent) - 1 - hits[-1])
    return out

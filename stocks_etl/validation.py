"""Validación honesta (López de Prado): Purged K-Fold, CPCV y Deflated Sharpe Ratio.

Cada evento i ocupa el intervalo [start_i, end_i] (end = toque de barrera, t1).
Purge: se quitan del train los eventos cuyo intervalo se solapa con el bloque de test.
Embargo: se quitan además los eventos que empiezan justo después del test, porque la
memoria del mercado (correlación serial) seguiría filtrando información.
"""
import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def _purged_train(starts: np.ndarray, ends: np.ndarray, test_idx: np.ndarray, embargo: np.timedelta64) -> np.ndarray:
    """Índices de train que no se solapan con ningún bloque contiguo de test (+ embargo)."""
    keep = np.ones(len(starts), dtype=bool)
    keep[test_idx] = False
    # bloques contiguos dentro del test (en CPCV puede haber varios)
    order = np.sort(test_idx)
    breaks = np.where(np.diff(order) > 1)[0]
    for block in np.split(order, breaks + 1):
        t0 = starts[block].min()
        t1 = ends[block].max()
        overlap = (starts <= t1 + embargo) & (ends >= t0)
        keep &= ~overlap
    return np.where(keep)[0]


def _embargo_span(starts: np.ndarray, embargo_pct: float) -> np.timedelta64:
    span = starts.max() - starts.min()
    return np.timedelta64(int(span / np.timedelta64(1, "s") * embargo_pct), "s")


class PurgedKFold:
    """K-Fold sin barajar, con purge + embargo. Los eventos deben venir ordenados por start."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, starts: Sequence, ends: Sequence) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        starts = np.asarray(starts, dtype="datetime64[ns]")
        ends = np.asarray(ends, dtype="datetime64[ns]")
        if np.any(np.diff(starts) < np.timedelta64(0, "ns")):
            raise ValueError("events must be sorted by start")
        embargo = _embargo_span(starts, self.embargo_pct)
        for test_idx in np.array_split(np.arange(len(starts)), self.n_splits):
            yield _purged_train(starts, ends, test_idx, embargo), test_idx


def cpcv_splits(starts: Sequence, ends: Sequence, n_groups: int = 6, k_test: int = 2, embargo_pct: float = 0.01):
    """Combinatorial Purged CV: C(N,k) splits; cada uno prueba en k grupos a la vez.

    Devuelve (train_idx, test_idx, test_groups, group_of) para reconstruir caminos.
    """
    starts = np.asarray(starts, dtype="datetime64[ns]")
    ends = np.asarray(ends, dtype="datetime64[ns]")
    embargo = _embargo_span(starts, embargo_pct)
    groups = np.array_split(np.arange(len(starts)), n_groups)
    group_of = np.empty(len(starts), dtype=int)
    for g, idx in enumerate(groups):
        group_of[idx] = g
    for test_groups in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[g] for g in test_groups])
        yield _purged_train(starts, ends, test_idx, embargo), test_idx, test_groups, group_of


def cpcv_paths(split_results: List[Tuple[Tuple[int, ...], dict]], n_groups: int, k_test: int) -> List[dict]:
    """Une las predicciones de cada split en φ = C(N,k)·k/N caminos completos.

    split_results: [(test_groups, {group: valor})]. El camino j de un grupo g usa el
    j-ésimo split (en orden) que contiene a g.
    """
    n_paths = math.comb(n_groups, k_test) * k_test // n_groups
    paths = [dict() for _ in range(n_paths)]
    seen = {g: 0 for g in range(n_groups)}
    for test_groups, per_group in split_results:
        for g in test_groups:
            paths[seen[g]][g] = per_group[g]
            seen[g] += 1
    return paths


def sharpe(returns: Sequence[float]) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """SR que un 'mono' obtendría como máximo tras n_trials intentos (benchmark del DSR)."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    return math.sqrt(var_sr) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
        + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(sr: float, n_trials: int, var_sr: float, skew: float, kurt: float, T: int) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    sr, var_sr: sin anualizar (misma frecuencia que T). kurt: curtosis de Pearson (normal = 3).
    DSR > 0.95 => la ventaja probablemente es real y no suerte de muchos intentos.
    """
    if T < 2:
        return float("nan")
    sr0 = expected_max_sharpe(n_trials, var_sr)
    denom = 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom)))


def min_sample_size(p1: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> int:
    """Nº de trades independientes para distinguir un win rate p1 de p0 (test bilateral)."""
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(power)
    n = ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
    return int(math.ceil(n))


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def log_trial(path: str, cfg: dict, sr: float, n_trades: int, as_of: str) -> Tuple[int, float]:
    """Registra el intento (una fila por configuración distinta) y devuelve (n_trials, var_sr).

    Re-ejecutar la misma configuración no cuenta como intento nuevo; cambiarla sí.
    """
    h = config_hash(cfg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"config_hash": h, "as_of": as_of, "sr_per_trade": sr, "n_trades": n_trades,
                         "config": json.dumps(cfg, sort_keys=True)}])
    if os.path.exists(path):
        prev = pd.read_csv(path)
        log = pd.concat([prev[prev["config_hash"] != h], row], ignore_index=True)
    else:
        log = row
    log.to_csv(path, index=False)
    srs = log["sr_per_trade"].astype(float)
    return int(len(log)), float(srs.var(ddof=1)) if len(srs) > 1 else 0.0


def read_trials(path: str) -> Tuple[int, float]:
    """(n_trials, var_sr) del registro sin añadir nada: para re-evaluar una config ya registrada."""
    if not os.path.exists(path):
        return 1, 0.0
    srs = pd.read_csv(path)["sr_per_trade"].astype(float)
    return max(int(len(srs)), 1), float(srs.var(ddof=1)) if len(srs) > 1 else 0.0

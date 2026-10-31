import math

import numpy as np
import pandas as pd
import pytest

from stocks_etl.features import ffd, min_ffd_d
from stocks_etl.labeling import triple_barrier
from stocks_etl.regime import hmm_filtered_regimes, hurst_rolling, ou_features
from stocks_etl.risk import (
    cornish_fisher_var, expected_max_losing_streak, kelly_fraction, max_losing_streak,
    streak_drawdown, var_es_t,
)
from stocks_etl.validation import (
    PurgedKFold, cpcv_paths, cpcv_splits, deflated_sharpe, expected_max_sharpe, log_trial, min_sample_size,
)


# --- validación -------------------------------------------------------------

def _events(n=300, hold=10):
    starts = pd.bdate_range("2020-01-01", periods=n)
    ends = starts + pd.offsets.BDay(hold)
    return starts.to_numpy(), ends.to_numpy()


def test_purged_kfold_no_overlap_and_embargo():
    starts, ends = _events()
    for tr, te in PurgedKFold(5, embargo_pct=0.02).split(starts, ends):
        t0, t1 = starts[te].min(), ends[te].max()
        # ningún evento de train se solapa con el intervalo de test
        assert not np.any((starts[tr] <= t1) & (ends[tr] >= t0))
        # embargo: nada de train empieza justo después del test
        span = starts.max() - starts.min()
        embargo = span * 0.02
        assert not np.any((starts[tr] > t1) & (starts[tr] <= t1 + embargo))
        assert len(tr) > 0


def test_purged_kfold_requires_sorted_events():
    starts, ends = _events()
    with pytest.raises(ValueError):
        list(PurgedKFold(3).split(starts[::-1], ends[::-1]))


def test_cpcv_paths_count_and_coverage():
    starts, ends = _events(240)
    results = []
    for tr, te, groups, group_of in cpcv_splits(starts, ends, n_groups=6, k_test=2):
        assert not set(tr) & set(te)
        results.append((groups, {g: g for g in groups}))
    assert len(results) == math.comb(6, 2)
    paths = cpcv_paths(results, 6, 2)
    assert len(paths) == 5  # φ = C(6,2)·2/6
    for p in paths:
        assert sorted(p) == list(range(6))  # cada camino recorre toda la historia


def test_deflated_sharpe_matches_bailey_lopez_de_prado_example():
    # Ejemplo numérico del paper (2014): SR anual 2.5, 100 intentos, Var[SR] anual 0.5,
    # 1250 observaciones diarias, asimetría -3, curtosis 10 -> SR0 ≈ 0.1132, DSR ≈ 0.90
    sr = 2.5 / math.sqrt(250)
    var_sr = 0.5 / 250
    assert expected_max_sharpe(100, var_sr) == pytest.approx(0.1132, abs=5e-4)
    assert deflated_sharpe(sr, 100, var_sr, -3, 10, 1250) == pytest.approx(0.90, abs=0.01)


def test_single_trial_dsr_is_probabilistic_sharpe_against_zero():
    assert expected_max_sharpe(1, 0.1) == 0.0
    assert deflated_sharpe(0.0, 1, 0.1, 0, 3, 500) == pytest.approx(0.5)


def test_min_sample_size_for_55pct_edge_is_about_800():
    assert 750 <= min_sample_size(0.55) <= 820


def test_trial_log_counts_distinct_configs(tmp_path):
    path = str(tmp_path / "trials.csv")
    assert log_trial(path, {"a": 1}, 0.1, 50, "2026-01-01")[0] == 1
    assert log_trial(path, {"a": 1}, 0.2, 60, "2026-01-02")[0] == 1  # misma config: no suma
    n, var = log_trial(path, {"a": 2}, 0.0, 70, "2026-01-03")
    assert n == 2 and var > 0


# --- etiquetado -------------------------------------------------------------

def test_triple_barrier_hits_take_profit_stop_and_time():
    close = np.array([100, 101, 102, 104, 103, 100, 97, 96, 96.5, 96.6, 96.7])
    out = triple_barrier(close, np.array([0, 4, 7]), np.array([1, 1, 1]),
                         np.array([0.03, 0.03, 0.03]), 1.0, 1.0, np.array([5, 5, 3]))
    assert list(out["barrier"]) == ["tp", "sl", "t"]
    assert list(out["end_pos"]) == [3, 6, 10]
    assert out["ret"].iloc[0] == pytest.approx(0.04)
    assert out["ret"].iloc[1] == pytest.approx(97 / 103 - 1)


def test_triple_barrier_short_side_profits_when_price_falls():
    close = np.array([100, 99, 96])
    out = triple_barrier(close, np.array([0]), np.array([-1]), np.array([0.03]), 1.0, 1.0, np.array([5]))
    assert out["barrier"].iloc[0] == "tp" and out["ret"].iloc[0] == pytest.approx(0.04)


def test_triple_barrier_marks_open_events():
    close = np.array([100, 100.5, 100.2])
    out = triple_barrier(close, np.array([1, 2]), np.array([1, 1]), np.array([0.05, 0.05]), 1, 1, np.array([5, 5]))
    assert list(out["barrier"]) == ["open", "open"]


# --- features y régimen -----------------------------------------------------

def test_ffd_makes_random_walk_stationary_with_partial_d():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, 2000)))
    d = min_ffd_d(x)
    assert 0 < d <= 1.0
    y = ffd(x, d)
    assert y.iloc[:10].isna().all() and y.iloc[-100:].notna().all()
    # conserva memoria: la serie diferenciada sigue correlacionada con la original
    assert y.corr(x) > 0.3 or d == 1.0


def test_ffd_with_d1_is_first_difference():
    x = pd.Series(np.arange(10, dtype=float) ** 2)
    y = ffd(x, 1.0)
    assert np.allclose(y.dropna().to_numpy(), x.diff().dropna().to_numpy())


def test_hurst_separates_trend_from_mean_reversion():
    rng = np.random.default_rng(1)
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 1500)))
    mr = pd.Series(np.zeros(1500))
    for t in range(1, 1500):
        mr[t] = 0.3 * mr[t - 1] + rng.normal()
    h_rw = hurst_rolling(rw, 200).dropna().mean()
    h_mr = hurst_rolling(mr, 200).dropna().mean()
    assert 0.4 < h_rw < 0.6
    assert h_mr < 0.25


def test_ou_half_life_recovered_on_simulated_process():
    rng = np.random.default_rng(2)
    theta = math.log(2) / 5  # vida media real = 5 días
    s = np.zeros(4000)
    for t in range(1, 4000):
        s[t] = s[t - 1] - theta * s[t - 1] + rng.normal(0, 0.01)
    # span muy largo: la EWM queda ~plana y la distancia a la media es el propio proceso OU
    hl = ou_features(pd.Series(s), span=10_000, window=2000)["half_life"].dropna()
    assert hl.iloc[-1] == pytest.approx(5, rel=0.3)


def test_hmm_filtered_probabilities_are_causal_and_ordered():
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.005, 400)
    stress = rng.normal(0, 0.03, 200)
    r = np.concatenate([calm, stress, calm])
    vol = pd.Series(r).rolling(20, min_periods=5).std().to_numpy()
    X = np.column_stack([r, vol])
    p = hmm_filtered_regimes(X)
    assert np.allclose(np.nansum(p, axis=1)[10:], 1.0)
    assert np.nanmean(p[450:600, 2]) > np.nanmean(p[100:400, 2])


# --- riesgo -----------------------------------------------------------------

def test_kelly_fraction():
    assert kelly_fraction(0.6, 1.0, frac=1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.6, 1.0, frac=0.5) == pytest.approx(0.1)
    assert kelly_fraction(0.4, 1.0) == 0.0  # sin ventaja, no se opera


def test_t_student_var_es_heavier_than_normal():
    v = var_es_t(0.01, 5, 0.01)
    assert v["es"] > v["var"] > 0
    # una t con muchos grados de libertad converge a la normal (VaR 99% ≈ 2.326σ)
    assert var_es_t(0.01, 1000, 0.01)["var"] == pytest.approx(0.02326, rel=0.01)


def test_cornish_fisher_penalises_negative_skew_and_fat_tails():
    base = cornish_fisher_var(0.01, 0, 0, 0.05)
    assert base == pytest.approx(0.01645, rel=1e-3)
    assert cornish_fisher_var(0.01, -1.0, 3.0, 0.01) > cornish_fisher_var(0.01, 0, 0, 0.01)


def test_losing_streak_examples_from_videos():
    # 40% de acierto en 100 trades -> racha de ~7; con 5% de riesgo es -30%, con 1% ~-6.8%
    k = expected_max_losing_streak(0.4, 100)
    assert k == pytest.approx(7.2, abs=0.1)
    assert streak_drawdown(7, 0.05) == pytest.approx(0.30, abs=0.01)
    assert streak_drawdown(7, 0.01) == pytest.approx(0.068, abs=0.001)
    assert max_losing_streak([1, 0, 0, 1, 0, 0, 0, 1]) == 3

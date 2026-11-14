import numpy as np
import pandas as pd
import pytest

from stocks_etl.regime import hmm_filtered_regimes_pit, prob_reach_state
from stocks_etl.regime_analytics import garch_diag, post_shock, transition_summary

A = np.array([[0.96, 0.04, 0.00], [0.03, 0.94, 0.03], [0.00, 0.05, 0.95]])


def test_probability_of_reaching_stress_matches_simulation():
    rng = np.random.default_rng(0)
    hits = 0
    n = 20000
    for _ in range(n):
        s = 1
        for _ in range(10):
            s = rng.choice(3, p=A[s])
            if s == 2:
                hits += 1
                break
    p = float(prob_reach_state(np.array([[0.0, 1.0, 0.0]]), A, 10)[0])
    assert p == pytest.approx(hits / n, abs=0.01)


def test_transition_summary_durations_and_age():
    reg = pd.Series([0] * 30 + [1] * 7)
    t = transition_summary(A, reg)
    assert t["current"] == "normal" and t["age_sessions"] == 7
    assert t["expected_duration_sessions"]["calm"] == pytest.approx(25.0)
    assert t["p_still_in_regime"]["10"] == pytest.approx(0.94 ** 10)


def test_returning_models_does_not_change_the_probabilities():
    rng = np.random.default_rng(1)
    n = 900
    vol = np.where(np.arange(n) % 300 < 200, 0.01, 0.03)
    r = rng.normal(0, vol)
    X = np.column_stack([r, pd.Series(r).rolling(20, min_periods=10).std().to_numpy()])
    dates = pd.bdate_range("2021-01-04", periods=n)
    plain = hmm_filtered_regimes_pit(X, dates, min_train=300, restarts=1)
    probs, models = hmm_filtered_regimes_pit(X, dates, min_train=300, restarts=1, return_models=True)
    assert np.allclose(np.nan_to_num(plain, nan=-1), np.nan_to_num(probs, nan=-1))
    assert models and all(np.allclose(m["transmat"].sum(axis=1), 1) for m in models)


def test_garch_long_run_falls_back_to_sample_vol_near_unit_persistence():
    ok = garch_diag({"persistence": 0.97, "sigma_long_run": 0.01, "nu": 6, "sigma_next": 0.02})
    assert ok["long_run_source"] == "garch" and ok["alert"] and ok["shock_half_life_sessions"] == pytest.approx(
        np.log(0.5) / np.log(0.97))
    near = garch_diag({"persistence": 0.999, "sigma_long_run": 0.04, "nu": 6, "sigma_next": 0.012},
                      sample_sigma=0.011)
    assert near["long_run_source"].startswith("sample") and near["vol_long_run_annual"] == pytest.approx(0.011 * 252 ** 0.5)


def test_post_shock_window():
    r = pd.Series(np.log1p([0.001] * 15 + [-0.045] + [0.002] * 3))
    s = post_shock(r)
    assert s["active"] and s["sessions_since_shock"] == 3
    assert not post_shock(pd.Series(np.log1p([-0.05] + [0.001] * 12)))["active"]

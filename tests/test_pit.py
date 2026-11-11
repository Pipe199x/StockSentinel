"""Point-in-time guarantees: nothing computed for date t may change when data after t is removed."""
import numpy as np
import pandas as pd
import pytest

from stocks_etl.features import make_features
from stocks_etl.meta_model import position_size
from stocks_etl.regime import refit_boundaries
from stocks_etl.risk import es_cap

CFG = {"pit_refit": True, "pit_min_train": 300, "pit_hmm_restarts": 1, "seed": 7}


def _prices(n=900, seed=3):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2016-01-04", periods=n, tz="UTC")
    rows = []
    for t in ("SPY", "AAA"):
        vol = np.where(np.arange(n) % 300 < 150, 0.01, 0.025)  # volatility regimes
        c = 100 * np.exp(np.cumsum(rng.normal(0, vol)))
        rows.append(pd.DataFrame({"date": d, "ticker": t, "open": c, "high": c * 1.01, "low": c * 0.99,
                                  "close": c, "volume": 1e6}))
    return pd.concat(rows, ignore_index=True)


def test_refit_boundaries_are_year_starts_after_burn_in():
    d = pd.bdate_range("2016-01-04", periods=900)
    b = refit_boundaries(d, 300)
    assert all(d[i].year != d[i - 1].year for i in b)
    assert all(i >= 300 for i in b) and b == sorted(b)


def test_features_do_not_change_when_future_is_removed():
    full = _prices()
    cut = full["date"].sort_values().unique()[700]
    f_full, _ = make_features(full, CFG)
    f_cut, _ = make_features(full[full["date"] <= cut], CFG)
    cols = ["sigma_garch", "garch_nu", "p_stress", "regime", "ffd_close", "mkt_p_stress"]
    a = f_full[f_full["date"] <= cut].set_index(["ticker", "date"])[cols].sort_index()
    b = f_cut.set_index(["ticker", "date"])[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6, atol=1e-9)


def test_burn_in_has_no_volatility_or_regime():
    f, _ = make_features(_prices(), CFG)
    first_year = f[f["date"] < pd.Timestamp("2017-01-01", tz="UTC")]
    assert first_year["sigma_garch"].isna().all() and first_year["p_stress"].isna().all()


def test_es_cap_binds_in_fat_tailed_high_vol_and_not_in_calm():
    cfg = {"max_es_loss": 0.01, "kelly_fraction": 0.5, "target_vol": 0.2, "k_tp": 1.5, "k_sl": 1.5}
    calm, limit = position_size(0.6, 0.004, 30.0, 3, cfg)
    assert limit == "kelly" and calm > 0
    wild, limit = position_size(0.6, 0.04, 3.0, 20, cfg)
    assert limit == "es"
    assert wild == pytest.approx(es_cap(0.04, 3.0, 20, 0.01))
    assert es_cap(float("nan"), 5, 5, 0.01) == 0.0

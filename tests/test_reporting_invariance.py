"""The advisor's measurement layer must never change a trade: same signals, same backtest."""
import numpy as np
import pandas as pd

from stocks_etl import advisor_report
from stocks_etl.meta_model import make_signals
from stocks_etl.paper import COLUMNS

CFG = {"strategies": ["momentum"], "barrier_mode": "live", "regime_source": "market", "momentum_horizon": 10,
       "n_splits": 3, "cpcv_groups": 4, "cpcv_k": 2, "mc_paths": 200, "k_tp": 1.5, "k_sl": 1.5, "seed": 3}


def _synthetic(n_tickers=3, n_days=650, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=n_days, tz="UTC")
    out, meta = [], {}
    for k in range(n_tickers):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n_days)))
        o = c * np.exp(rng.normal(0, 0.004, n_days))
        df = pd.DataFrame({"date": days, "ticker": f"T{k}", "open": o, "high": np.maximum(o, c) * 1.006,
                           "low": np.minimum(o, c) * 0.994, "close": c, "volume": 1e6, "dividends": 0.0})
        lp = np.log(df["close"])
        df["log_ret_1"], df["log_ret_5"], df["log_ret_20"] = lp.diff(), lp.diff(5), lp.diff(20)
        df["rv_10"], df["rv_20"] = df["log_ret_1"].rolling(10).std(), df["log_ret_1"].rolling(20).std()
        df["sigma_garch"], df["garch_nu"] = df["rv_20"], 6.0
        df["ffd_close"] = lp - lp.rolling(50).mean()
        df["hurst"] = rng.uniform(0.45, 0.70, n_days)
        df["zscore"] = rng.normal(0, 1.2, n_days)
        df["half_life"] = rng.uniform(3, 30, n_days)
        df["vol_z"], df["beta_spy"] = rng.normal(0, 1, n_days), 1.0
        df["regime"] = rng.choice([0, 1], n_days)
        df["p_calm"], df["p_normal"], df["p_stress"] = 0.5, 0.45, 0.05
        df["mkt_regime"], df["mkt_p_stress"], df["mkt_p_stress_10"] = df["regime"], 0.05, 0.1
        out.append(df.iloc[25:])
        meta[f"T{k}"] = {"ffd_d": 0.4, "garch": {"sigma_next": 0.015, "nu": 6.0, "mu": 0.0, "persistence": 0.97,
                                                 "sigma_long_run": 0.014}}
    return pd.concat(out, ignore_index=True), meta


def test_measurement_layer_changes_no_signal_and_no_backtest_number(tmp_path):
    feats, meta = _synthetic()
    s1, r1 = make_signals(feats, meta, CFG)
    s2, r2, internals = make_signals(feats, meta, CFG, return_internals=True)
    pd.testing.assert_frame_equal(s1, s2)
    assert r1["meta"] == r2["meta"] and r1["cpcv_path_sr"] == r2["cpcv_path_sr"] and r1["config_hash"] == r2["config_hash"]

    before = (s2.copy(), dict(r2["meta"]))
    prices = feats[["date", "ticker", "open", "high", "low", "close", "volume", "dividends"]]
    ledger = pd.DataFrame(columns=COLUMNS)
    adv = advisor_report.build(prices, feats, meta, internals, r2, s2, ledger, {}, {"quant": CFG}, str(tmp_path))
    pd.testing.assert_frame_equal(s2, before[0])                 # no mutation of the published signals
    assert r2["meta"] == before[1]
    assert set(adv) == {"meta", "validation", "risk", "paper"}
    assert adv["meta"]["calendar"]["sessions"] > 0 and adv["meta"]["effective_n"] > 0
    assert len(adv["validation"]["stability"]["runs"]) == 3
    for col in ("thesis", "risk_to_stop_pct_equity", "es_trade_pct_equity", "market_p_stress_10"):
        assert col in s2.columns

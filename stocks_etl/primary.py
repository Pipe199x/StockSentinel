"""Señal primaria: el régimen decide la estrategia.

Regla fija, escrita antes de mirar resultados (no se optimiza: cada ajuste sería un
intento más que el Deflated Sharpe debe castigar):
  - Régimen de estrés del mercado    -> plano. No se opera.
    (regime_source: market usa el HMM del benchmark; asset, el del propio activo)
  - Hurst >= 0.5 (persistente)       -> momentum: a favor del retorno de 20 días si el
                                        precio está del mismo lado de su valor justo.
  - Hurst <  0.5 (antipersistente)   -> reversión a la media OU: |z| >= z_entry y vida
                                        media corta (operable).
"""
import numpy as np
import pandas as pd

STRATEGY_NAMES = {0: "none", 1: "momentum", 2: "mean_reversion"}


def primary_side(f: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    z_entry = float(cfg.get("z_entry", 1.5))
    max_hl = float(cfg.get("max_half_life", 20))
    mom_h = int(cfg.get("momentum_horizon", 10))
    t_min, t_cap = int(cfg.get("t_max_min", 3)), int(cfg.get("t_max_cap", 20))

    regime_col = "mkt_regime" if cfg.get("regime_source", "asset") == "market" else "regime"
    needed = ["hurst", "zscore", "half_life", "log_ret_20", "sigma_garch", regime_col]
    ok = f[needed].notna().all(axis=1) & np.isfinite(f["sigma_garch"])
    stress = f[regime_col] == 2

    trend = ok & ~stress & (f["hurst"] >= 0.5) & (np.sign(f["log_ret_20"]) == np.sign(f["zscore"]))
    revert = ok & ~stress & (f["hurst"] < 0.5) & (f["half_life"] <= max_hl) & (f["zscore"].abs() >= z_entry)

    side = np.zeros(len(f), dtype=int)
    side[trend.to_numpy()] = np.sign(f.loc[trend, "log_ret_20"]).astype(int)
    side[revert.to_numpy()] = -np.sign(f.loc[revert, "zscore"]).astype(int)

    strategy = np.where(trend, 1, np.where(revert, 2, 0))
    hl = f["half_life"].fillna(t_cap).clip(lower=t_min, upper=t_cap).round()
    t_max = np.where(strategy == 2, hl, mom_h).astype(int)
    return pd.DataFrame({"side": side, "strategy": strategy, "t_max": t_max}, index=f.index)

import pandas as pd
import numpy as np

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Espera columnas: date,ticker,close,volume
# Devuelve DF con features + target ret_t1

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["ticker", "date"]).copy()
    out = []
    for tkr, dfg in df.groupby("ticker", sort=False):
        dfg = dfg.copy()
        dfg["ret_1d"] = dfg["close"].pct_change(1)
        dfg["ret_5d"] = dfg["close"].pct_change(5)
        dfg["vol_roll_10"] = dfg["volume"].rolling(10, min_periods=5).mean()
        dfg["sma_5"] = dfg["close"].rolling(5, min_periods=3).mean()
        dfg["sma_20"] = dfg["close"].rolling(20, min_periods=10).mean()
        dfg["rsi_14"] = _rsi(dfg["close"], 14)
        dfg["lag1_close"] = dfg["close"].shift(1)
        dfg["lag1_volume"] = dfg["volume"].shift(1)
        # target como retorno del siguiente día
        dfg["ret_t1"] = dfg["close"].shift(-1) / dfg["close"] - 1.0
        out.append(dfg)
    feats = pd.concat(out, ignore_index=True)
    # Quita filas muy tempranas sin features/target
    feats = feats.dropna(subset=["ret_1d", "lag1_close"])  # pero ret_t1 del último día será NaN
    return feats

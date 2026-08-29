from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Conjunto explícito de features permitidas (presentes tras features.make_features)
ALLOWED_FEATURES: List[str] = [
    "ret_1d",
    "ret_5d",
    "vol_roll_10",
    "sma_5",
    "sma_20",
    "rsi_14",
    "lag1_close",
    "lag1_volume",
    "close",    # opcional como nivel
    "volume",   # opcional como nivel
    # IMPORTANTE: NO incluir "adj_close" (puede ser todo NaN si auto_adjust=True)
]

@dataclass
class XGBParams:
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


def _fit_predict_last(dfg: pd.DataFrame, params: XGBParams, backtest_days: int = 30) -> Dict:
    # columnas de entrada: intersección entre columnas del DF y ALLOWED_FEATURES
    feat_cols = [c for c in ALLOWED_FEATURES if c in dfg.columns]

    if not feat_cols:
        return {"pred_ret_t1": None, "mae_backtest": None}

    # última fila con features válidas
    d_valid = dfg.dropna(subset=feat_cols)
    if d_valid.empty:
        return {"pred_ret_t1": None, "mae_backtest": None}
    last_row = d_valid.iloc[[-1]]

    # datos con target disponible
    train_df = dfg.dropna(subset=feat_cols + ["ret_t1"]).copy()
    if len(train_df) < 50:
        return {"pred_ret_t1": None, "mae_backtest": None}

    X = train_df[feat_cols]
    y = train_df["ret_t1"]

    model = XGBRegressor(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        random_state=params.random_state,
        n_jobs=1,
        objective="reg:squarederror",
    )

    model.fit(X, y)

    # backtest simple: últimos N días
    if backtest_days > 0 and len(train_df) > backtest_days:
        X_bt = X.iloc[-backtest_days:]
        y_bt = y.iloc[-backtest_days:]
        y_hat_bt = model.predict(X_bt)
        mae_bt = float(mean_absolute_error(y_bt, y_hat_bt))
    else:
        mae_bt = None

    pred = float(model.predict(last_row[feat_cols])[0])

    return {
        "pred_ret_t1": pred,
        "mae_backtest": mae_bt,
    }


def _next_trading_day(as_of: pd.Timestamp, cal_name: Optional[str]) -> str:
    """Primera sesión bursátil estrictamente posterior a as_of (fin de semana/festivos excluidos)."""
    start = (as_of + pd.Timedelta(days=1)).date()
    end = (as_of + pd.Timedelta(days=14)).date()
    if cal_name:
        try:
            import pandas_market_calendars as mcal
            sched = mcal.get_calendar(cal_name).schedule(start_date=start, end_date=end)
            if not sched.empty:
                return sched.index[0].strftime("%Y-%m-%d")
        except Exception:
            pass
    # Fallback: siguiente día hábil (lun-vie)
    d = as_of + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def predict_next_day(df_feats: pd.DataFrame, backtest_days: int = 30, params: Optional[XGBParams] = None, cal_name: Optional[str] = None) -> pd.DataFrame:
    params = params or XGBParams()
    results = []
    for tkr, dfg in df_feats.groupby("ticker", sort=False):
        dfg = dfg.sort_values("date")
        metrics = _fit_predict_last(dfg, params, backtest_days)
        as_of_ts = dfg["date"].max()
        as_of = as_of_ts.strftime("%Y-%m-%d")
        target_date = _next_trading_day(as_of_ts, cal_name)
        # yfinance puede añadir una fila final con NaN; usar el último cierre válido
        closes = dfg["close"].dropna() if "close" in dfg.columns else pd.Series(dtype=float)
        last_close = float(closes.iloc[-1]) if not closes.empty else None
        pred_ret = metrics["pred_ret_t1"]
        pred_close = float(last_close * (1 + pred_ret)) if (pred_ret is not None and last_close is not None) else None
        results.append({
            "ticker": tkr,
            "as_of": as_of,
            "target_date": target_date,
            "last_close": float(last_close) if last_close is not None else None,
            "pred_ret_t1": pred_ret,
            "pred_close_t1": pred_close,
            "mae_backtest_30d": metrics["mae_backtest"],
            # "model": "xgb_v1",  # ya no la emitimos; se elimina en write_predictions_csv si existiera
        })
    return pd.DataFrame(results)

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

@dataclass
class XGBParams:
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


def _fit_predict_last(dfg: pd.DataFrame, params: XGBParams, backtest_days: int = 30) -> Dict:
    # columnas de entrada (excluye obvias y target)
    drop_cols = {"date", "ticker", "ret_t1"}
    feat_cols = [c for c in dfg.columns if c not in drop_cols]

    # fila para predecir (última con features válidas)
    last_row = dfg.dropna(subset=feat_cols).iloc[[-1]]

    # datos con target disponible
    train_df = dfg.dropna(subset=feat_cols + ["ret_t1"]).copy()
    if len(train_df) < 50:
        return {
            "pred_ret_t1": None,
            "mae_backtest": None,
        }

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


def predict_next_day(df_feats: pd.DataFrame, backtest_days: int = 30, params: Optional[XGBParams] = None) -> pd.DataFrame:
    params = params or XGBParams()
    results = []
    for tkr, dfg in df_feats.groupby("ticker", sort=False):
        dfg = dfg.sort_values("date")
        metrics = _fit_predict_last(dfg, params, backtest_days)
        as_of = dfg["date"].max()
        last_close = dfg.loc[dfg.index[-1], "close"]
        pred_ret = metrics["pred_ret_t1"]
        pred_close = float(last_close * (1 + pred_ret)) if pred_ret is not None else None
        results.append({
            "ticker": tkr,
            "as_of": as_of,
            "last_close": float(last_close),
            "pred_ret_t1": pred_ret,
            "pred_close_t1": pred_close,
            "mae_backtest_30d": metrics["mae_backtest"],
            "model": "xgb_v1",
        })
    return pd.DataFrame(results)

"""Publishers for the GitHub Pages data folder (docs/data/).

The dashboard (docs/index.html) fetches these static JSON/CSV files directly;
GitHub Pages serves them with permissive CORS, so no hosted API is needed.
"""
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_prices_latest_json(df: pd.DataFrame, path: str, window_days: int = 30) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    recent = df[df["date"] >= cutoff]
    series = {}
    for tkr, dfg in recent.groupby("ticker"):
        dfg = dfg.sort_values("date")
        # yfinance failed downloads arrive as all-NaN rows; they say nothing
        dfg = dfg[dfg["close"].notna()]
        series[tkr] = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "close": round(float(c), 4),
                "adj_close": (round(float(a), 4) if pd.notna(a) else None),
                "volume": (int(v) if pd.notna(v) else None),
            }
            for d, c, a, v in zip(dfg["date"], dfg["close"], dfg["adj_close"], dfg["volume"])
        ]
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "series": series,
    }
    _ensure_dir(path)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_predictions_latest_json(preds: pd.DataFrame, path: str) -> str:
    preds = preds.drop(columns=["model"], errors="ignore")
    _ensure_dir(path)
    records = json.loads(preds.to_json(orient="records"))
    Path(path).write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_prices_history_csv(df: pd.DataFrame, path: str) -> str:
    _ensure_dir(path)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path


def append_predictions_history(preds: pd.DataFrame, path: str) -> str:
    preds = preds.drop(columns=["model"], errors="ignore")
    _ensure_dir(path)
    if os.path.exists(path):
        prev = pd.read_csv(path)
        combined = pd.concat([prev, preds], ignore_index=True)
    else:
        combined = preds
    combined = combined.drop_duplicates(subset=["ticker", "as_of"], keep="last")
    combined = combined.sort_values(["as_of", "ticker"])
    combined.to_csv(path, index=False)
    return path


def _clean(obj):
    """NaN/inf -> None para que el JSON sea estricto (los navegadores no aceptan NaN)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def write_json(obj, path: str) -> str:
    _ensure_dir(path)
    Path(path).write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_signals_latest_json(signals: pd.DataFrame, path: str) -> str:
    return write_json(signals.to_dict(orient="records"), path)


def append_signals_history(signals: pd.DataFrame, path: str) -> str:
    """Una fila por (ticker, as_of): permite medir después cómo terminó cada señal."""
    cols = ["ticker", "as_of", "last_close", "regime", "market_regime", "strategy", "side", "p_success",
            "kelly_size", "tp", "sl", "t_max_date", "sigma_garch", "var_95", "es_99"]
    _ensure_dir(path)
    new = signals[cols]
    if os.path.exists(path):
        combined = pd.concat([pd.read_csv(path), new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=["ticker", "as_of"], keep="last").sort_values(["as_of", "ticker"])
    combined.to_csv(path, index=False)
    return path

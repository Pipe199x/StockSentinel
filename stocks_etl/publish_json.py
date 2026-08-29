"""Publishers for the GitHub Pages data folder (docs/data/).

The Lovable frontend fetches these static JSON/CSV files directly; GitHub
Pages serves them with permissive CORS, so no hosted API is needed.
"""
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_prices_latest_json(df: pd.DataFrame, path: str, window_days: int = 30) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    recent = df[df["date"] >= cutoff]
    series = {}
    for tkr, dfg in recent.groupby("ticker"):
        dfg = dfg.sort_values("date")
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
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_predictions_latest_json(preds: pd.DataFrame, path: str) -> str:
    preds = preds.drop(columns=["model"], errors="ignore")
    _ensure_dir(path)
    records = json.loads(preds.to_json(orient="records"))
    Path(path).write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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

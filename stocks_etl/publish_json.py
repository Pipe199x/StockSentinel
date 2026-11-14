"""Publishers for the dashboard's data folder (docs/data/).

The dashboard fetches these static JSON/CSV files at runtime. Cloudflare Pages serves them
behind Cloudflare Access (owner only): Massive's license forbids publishing its data.
"""
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_prices_latest_json(df: pd.DataFrame, path: str, window_days: int = 30, now: datetime = None) -> str:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)
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
        "as_of": now.isoformat(),
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


# === Paywall: the public files carry no pick identity; the full ones go to docs/data/premium/ ===
# (served only behind the Cloudflare Pages Function in functions/data/_middleware.ts).
PUBLIC_PICK_FIELDS = ("as_of", "side", "market_regime", "market_p_stress")  # exit date and setup are paid too


def redact_signals(records: list) -> dict:
    """Public view of signals_latest: today's picks without ticker, levels, odds or size.

    Idle assets are only counted: naming them would reveal the picks by elimination."""
    active = [r for r in records if (r.get("kelly_size") or 0) > 0]
    first = records[0] if records else {}
    return {
        "as_of": first.get("as_of"),
        "market_regime": first.get("market_regime"),
        "market_p_stress": first.get("market_p_stress"),
        "n_assets": len(records),
        "n_idle": len(records) - len(active),
        "picks": [{k: r.get(k) for k in PUBLIC_PICK_FIELDS} for r in active],
    }


def redact_paper(paper: dict) -> dict:
    """Public view of paper_summary: totals and curve stay; open positions lose ticker, size and prices."""
    out = {k: v for k, v in paper.items() if k not in ("positions", "rescore")}
    open_rows = paper.get("positions") or []
    out["positions"] = [{"side": r.get("side"), "status": r.get("status")} for r in open_rows]
    out["invested"] = float(sum((r.get("size") or 0) for r in open_rows))
    return out


def write_public_and_premium(obj, docs_dir: str, name: str, redact=None) -> str:
    """Full file → docs/data/premium/<name>; redacted copy (or nothing) → docs/data/<name>."""
    full = write_json(obj, os.path.join(docs_dir, "premium", name))
    public = os.path.join(docs_dir, name)
    if redact is not None:
        write_json(redact(json.loads(Path(full).read_text(encoding="utf-8"))), public)
    elif os.path.exists(public):
        os.remove(public)  # premium-only file: no public copy left behind
    return full


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

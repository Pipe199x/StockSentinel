"""Risk-free rate for excess-return Sharpe (advisor, 2 Nov video: Sharpe = (return - risk-free)
/ volatility). FRED series DTB3, the 3-month Treasury bill (public CSV, no key, not Massive
data). Refreshed weekly by deep-weekly.yml into docs/data/history/risk_free.csv; when the file
is missing the pipeline uses 0 and says so.

    python -m etl_common.risk_free docs/data/history/risk_free.csv
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"


def fetch_dtb3(timeout: int = 60) -> pd.DataFrame:
    import requests
    r = requests.get(FRED_CSV, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "rate"]
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")          # FRED marks holidays with "."
    return df.dropna().reset_index(drop=True)


def load_daily_rf(path: str) -> pd.Series:
    """Daily risk-free return indexed by date (annual % / 100 / 252), forward-filled; empty if missing."""
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    s = pd.Series(df["rate"].astype(float).to_numpy() / 100.0 / 252.0, index=pd.to_datetime(df["date"]))
    return s.sort_index()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/data/history/risk_free.csv"
    data = fetch_dtb3()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    data.to_csv(out, index=False)
    print(f"[RISK-FREE] DTB3 {len(data)} rows {data['date'].iloc[0]}..{data['date'].iloc[-1]} -> {out}")

"""Bulk, resumable download for the Massive research month.

    python -m research.massive.download [--start YYYY-MM-DD] [--max-per-min 5]

Saves to data/massive/ (gitignored):
  grouped/<YYYY-MM-DD>.csv.gz   every US stock ticker's daily bar for that session
  tickers.csv                   reference data: common stocks + ETFs, active AND delisted
Re-running skips sessions already on disk, so an interrupted run just continues.
Use --max-per-min 5 to test the code with a free Basic key before paying.
"""
import argparse
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from research.massive.client import MassiveClient, load_api_key

OUT = Path("data/massive")


def sessions(start: str, end: str) -> list:
    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return [d.strftime("%Y-%m-%d") for d in sched.index]


def download_grouped(client: MassiveClient, start: str, end: str, out: Path = OUT) -> int:
    gdir = out / "grouped"
    gdir.mkdir(parents=True, exist_ok=True)
    todo = [d for d in sessions(start, end) if not (gdir / f"{d}.csv.gz").exists()]
    print(f"[grouped] {len(todo)} sessions to download")
    for i, d in enumerate(todo, 1):
        rows = client.grouped_daily(d)
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"[grouped] {d}: no data (holiday or outside plan history)")
            continue
        df = df.rename(columns={"T": "ticker", "o": "open", "h": "high", "l": "low", "c": "close",
                                "v": "volume", "vw": "vwap", "n": "transactions", "t": "ts"})
        df.insert(0, "date", d)
        tmp = gdir / f"{d}.csv.gz.tmp"
        df.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(gdir / f"{d}.csv.gz")  # atomic: a crash never leaves a half file
        if i % 50 == 0 or i == len(todo):
            print(f"[grouped] {i}/{len(todo)} ({d}, {len(df)} tickers)")
    return len(todo)


def download_tickers(client: MassiveClient, out: Path = OUT) -> pd.DataFrame:
    rows = []
    for active in (True, False):
        for typ in ("CS", "ETF"):
            for r in client.tickers(active=active, ticker_type=typ):
                rows.append({k: r.get(k) for k in ("ticker", "name", "type", "active", "primary_exchange",
                                                   "composite_figi", "list_date", "delisted_utc")})
            print(f"[tickers] active={active} type={typ}: {len(rows)} rows so far")
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tickers.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    # Developer plan history is ~10 years; a week of margin avoids "not authorized" on the oldest day
    ten_years = (pd.Timestamp.today() - pd.DateOffset(years=10) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    ap.add_argument("--start", default=ten_years)
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--max-per-min", type=int, default=None, help="5 for the free Basic tier")
    ap.add_argument("--skip-tickers", action="store_true")
    a = ap.parse_args()
    client = MassiveClient(load_api_key(), max_per_min=a.max_per_min)
    if not a.skip_tickers:
        download_tickers(client)
    download_grouped(client, a.start, a.end)


if __name__ == "__main__":
    main()

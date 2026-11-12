"""yfinance download helpers, kept only for research baselines (research/replay, M0).
The live pipeline uses Massive only (stocks_etl.stocks_etl.load_prices)."""
import random
import time

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from tenacity import retry, wait_exponential_jitter, stop_after_attempt


@retry(wait=wait_exponential_jitter(initial=1, max=30), stop=stop_after_attempt(5))
def _download(symbols, start, end, interval, auto_adjust, actions, threads):
    return yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=actions,
        group_by="ticker",
        # threads=1 is truthy and still starts yfinance's worker pool; concurrent
        # writes to its SQLite cache cause intermittent 'database is locked'.
        threads=bool(threads) and threads > 1,
        progress=False,
    )


def _missing_symbols(df_raw, symbols):
    """Tickers with no valid Close in the raw yfinance frame."""
    missing = []
    for s in symbols:
        if isinstance(df_raw.columns, pd.MultiIndex):
            if s not in df_raw.columns.get_level_values(0):
                missing.append(s)
                continue
            dfi = df_raw[s]
        else:
            dfi = df_raw
        if "Close" not in dfi.columns or dfi["Close"].notna().sum() == 0:
            missing.append(s)
    return missing


def _download_with_retry(symbols, start, end, interval, auto_adjust, actions, threads, attempts=4):
    """Batch download, then re-download failed tickers one by one with backoff.

    yf.download does not raise when a ticker fails (it returns an all-NaN column),
    so tenacity on _download alone never retries those transient failures.
    """
    df_raw = _download(symbols, start, end, interval, auto_adjust, actions, threads)
    if not isinstance(df_raw.columns, pd.MultiIndex):
        df_raw = pd.concat({symbols[0]: df_raw}, axis=1) if len(symbols) == 1 else df_raw

    missing = _missing_symbols(df_raw, symbols)
    for attempt in range(1, attempts + 1):
        if not missing:
            break
        delay = 2 ** attempt + random.uniform(0, 1)
        print(f"[RETRY] attempt {attempt}/{attempts} for {missing} in {delay:.1f}s")
        time.sleep(delay)
        for s in missing:
            try:
                dfs = _download([s], start, end, interval, auto_adjust, actions, threads)
            except Exception as e:
                print(f"[RETRY] {s}: {e!r}")
                continue
            if not isinstance(dfs.columns, pd.MultiIndex):
                dfs = pd.concat({s: dfs}, axis=1)
            if _missing_symbols(dfs, [s]):
                continue
            df_raw = df_raw.drop(columns=s, level=0, errors="ignore")
            df_raw = pd.concat([df_raw, dfs], axis=1)
        missing = _missing_symbols(df_raw, symbols)
    return df_raw


def _normalize(df_raw, symbols, auto_adjust, cal_name):
    """Normaliza a un esquema uniforme y concatena todo en un solo DataFrame."""
    records = []
    cal = mcal.get_calendar(cal_name) if cal_name else None

    for s in symbols:
        dfi = df_raw[s] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw
        if dfi.empty:
            continue

        # reset index y estandarizar nombres
        dfi = dfi.reset_index()
        dfi.columns = [str(c).lower() for c in dfi.columns]

        # detectar columna temporal de forma robusta (date/datetime/primera datetime)
        dt_col = None
        for cand in ("date", "datetime"):
            if cand in dfi.columns:
                dt_col = cand
                break
        if dt_col is None:
            for c in dfi.columns:
                if pd.api.types.is_datetime64_any_dtype(dfi[c]):
                    dt_col = c
                    break
        if dt_col is None and pd.api.types.is_datetime64_any_dtype(dfi.index):
            dfi = dfi.reset_index().rename(columns={"index": "date"})
            dt_col = "date"
        if dt_col is None:
            raise ValueError(f"No se encontró columna de tiempo. Columnas: {list(dfi.columns)}")

        # renombrar OHLC/acciones
        rename = {"adj close": "adj_close", "stock splits": "stock_splits"}
        dfi = dfi.rename(columns=rename)

        # crear columna 'date' UTC y eliminar la original si es distinta
        dfi["date"] = pd.to_datetime(dfi[dt_col], utc=True)
        if dt_col != "date":
            dfi = dfi.drop(columns=[dt_col])

        dfi["ticker"] = s

        # consistencia de columnas
        if auto_adjust and "adj_close" not in dfi.columns:
            dfi["adj_close"] = pd.NA
        for c in ("dividends", "stock_splits"):
            if c not in dfi.columns:
                dfi[c] = 0.0

        # filtrar a días de mercado (diario)
        if cal is not None and "date" in dfi.columns:
            try:
                start_d = dfi["date"].min().date()
                end_d = dfi["date"].max().date()
                sched = cal.schedule(start_date=start_d, end_date=end_d)
                valid = pd.Index(
                    pd.to_datetime(sched.index)
                    .tz_localize("America/New_York")
                    .tz_convert("UTC")
                    .normalize()
                )
                dfi = dfi[dfi["date"].dt.normalize().isin(valid)]
            except Exception:
                pass

        records.append(
            dfi[
                [
                    "date",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "dividends",
                    "stock_splits",
                ]
            ]
        )

    out = (
        pd.concat(records, ignore_index=True).sort_values(["ticker", "date"])
        if records
        else pd.DataFrame()
    )
    return out

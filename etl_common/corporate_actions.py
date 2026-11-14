"""Cash dividends on the same scale as the split-adjusted prices.

Massive's daily bars (adjusted=true) are adjusted for splits only. Without dividends, an
ex-dividend drop looks like a loss to a long position and a gain to a short one: about 0.9%
a quarter for XOM, against an average edge of a few tenths of a percent per trade.

Massive quotes `cash_amount` per share as of the ex-date, i.e. before any later split, while
the prices are divided by every later split. So each dividend is converted to the price scale:
    cash_adj = cash_amount * prod(split_from / split_to) over splits executed after the ex-date
(NVDA: a dividend paid before the 2024 1->10 split is divided by 10, like the prices.)

The rule applied in the backtest labels AND the paper ledger (exact parity): barriers stay on
the split-adjusted prices; a trade receives (long) or pays (short) the dividends whose ex-date
falls in (entry day, exit day]. Buying at the open of the ex-date does not earn the dividend;
selling on the ex-date does.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

DIV_COLUMNS = ["ticker", "ex_date", "pay_date", "cash_amount", "split_factor", "cash_adj",
               "dividend_type", "frequency", "id"]


def fetch_dividends(client, symbols: Iterable[str], start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(dividends, splits) from Massive for every symbol since `start`."""
    divs, splits = [], []
    for t in symbols:
        for d in client.dividends(t, start):
            if (d.get("currency") or "USD") != "USD" or not d.get("ex_dividend_date"):
                continue
            divs.append({"ticker": t, "ex_date": d["ex_dividend_date"], "pay_date": d.get("pay_date"),
                         "cash_amount": float(d.get("cash_amount") or 0.0), "dividend_type": d.get("dividend_type"),
                         "frequency": d.get("frequency"), "id": d.get("id")})
        for s in client.splits(t, start):
            if s.get("execution_date") and s.get("split_from") and s.get("split_to"):
                splits.append({"ticker": t, "execution_date": s["execution_date"],
                               "split_from": float(s["split_from"]), "split_to": float(s["split_to"])})
    return (pd.DataFrame(divs, columns=[c for c in DIV_COLUMNS if c not in ("split_factor", "cash_adj")]),
            pd.DataFrame(splits, columns=["ticker", "execution_date", "split_from", "split_to"]))


def split_adjusted_cash(divs: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Add split_factor and cash_adj (dividend on the split-adjusted price scale)."""
    out = divs.copy()
    if out.empty:
        return pd.DataFrame(columns=DIV_COLUMNS)
    factor = np.ones(len(out))
    if not splits.empty:
        for i, (t, ex) in enumerate(zip(out["ticker"], out["ex_date"])):
            later = splits[(splits["ticker"] == t) & (splits["execution_date"] > ex)]
            factor[i] = float(np.prod(later["split_from"] / later["split_to"])) if len(later) else 1.0
    out["split_factor"] = factor
    out["cash_adj"] = out["cash_amount"].astype(float) * factor
    return out[DIV_COLUMNS].sort_values(["ticker", "ex_date"]).reset_index(drop=True)


def attach_dividends(prices: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    """Fill prices['dividends'] with cash_adj on each ex-date row (0 elsewhere)."""
    out = prices.copy()
    out["dividends"] = 0.0
    if divs is None or divs.empty:
        return out
    day = pd.to_datetime(out["date"])
    day = (day.dt.tz_convert("UTC").dt.tz_localize(None) if day.dt.tz is not None else day).dt.normalize()
    per_day = divs.assign(ex=pd.to_datetime(divs["ex_date"]).dt.normalize()).groupby(["ticker", "ex"])["cash_adj"].sum()
    key = pd.MultiIndex.from_arrays([out["ticker"], day])
    out["dividends"] = per_day.reindex(key).fillna(0.0).to_numpy()
    return out


def load_dividends_table(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={"ticker": str, "ex_date": str, "pay_date": str, "id": str})


def refresh_dividends(prices: pd.DataFrame, symbols, path: str, client=None,
                      offline: bool = False) -> Tuple[pd.DataFrame, dict]:
    """Prices with dividends attached + a status dict. Refetches the whole table each run (it
    heals itself); if the API fails, or offline, reuses the last published table and says so."""
    start = str(pd.to_datetime(prices["date"]).min().date())
    status = {"source": None, "rows": 0, "error": None}
    table = None
    if not offline:
        try:
            if client is None:
                from etl_common.massive import MassiveClient, load_api_key
                client = MassiveClient(load_api_key())
            divs, splits = fetch_dividends(client, symbols, start)
            table = split_adjusted_cash(divs, splits)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            table.to_csv(path, index=False)
            status["source"] = "massive"
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - a correction, never a reason to fail the run
            # (MassiveClient raises SystemExit on 401/403, e.g. if the plan stopped serving the endpoint)
            status["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    if table is None:
        table = load_dividends_table(path)
        status["source"] = "last_published" if table is not None else "none"
    status["rows"] = 0 if table is None else int(len(table))
    return attach_dividends(prices, table), status

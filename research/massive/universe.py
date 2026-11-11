"""Point-in-time universe + minute-bar resolver for the Massive research run.

The live pipeline trades 21 stocks that are alive *today*: a backtest on them never
meets a company that went bankrupt or was delisted (survivorship bias). Here the
universe is rebuilt every month from what was knowable then: the top-N US common
stocks by trailing dollar volume, including tickers that later disappeared.
"""
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from research.massive.client import MassiveClient

DATA = Path("data/massive")
ALWAYS = ["SPY", "QQQ", "GLD", "TLT"]  # same ETFs as the live universe; SPY drives the market regime
COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def load_grouped(data: Path = DATA, tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
    keep = set(tickers) if tickers is not None else None
    parts = []
    for f in sorted((data / "grouped").glob("*.csv.gz")):
        d = pd.read_csv(f, usecols=COLS, dtype={"ticker": str})
        if keep is not None:
            d = d[d["ticker"].isin(keep)]
        parts.append(d)
    if not parts:
        raise SystemExit(f"No grouped files in {data / 'grouped'}: run research.massive.download first")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def split_reused_tickers(df: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
    """A symbol that disappears for > max_gap sessions and comes back is usually a different
    company reusing it; give each stretch its own name (TICKER~1, ~2...) so prices never splice."""
    sess = {d: i for i, d in enumerate(sorted(df["date"].unique()))}
    idx = df["date"].map(sess).to_numpy()
    df = df.assign(_i=idx)
    gap = df.groupby("ticker")["_i"].diff().fillna(1).to_numpy() > max_gap
    seg = pd.Series(gap.astype(int), index=df.index).groupby(df["ticker"]).cumsum()
    df["ticker"] = np.where(seg > 0, df["ticker"] + "~" + seg.astype(str), df["ticker"])
    return df.drop(columns="_i")


def common_stocks(data: Path = DATA) -> set:
    t = pd.read_csv(data / "tickers.csv", dtype=str)
    return set(t.loc[t["type"] == "CS", "ticker"])


def pit_universe(df: pd.DataFrame, cs: set, top_n: int = 100, min_price: float = 5.0,
                 min_history: int = 300, window: int = 60) -> pd.DataFrame:
    """Monthly membership: on each month's last session rank common stocks by their median
    dollar volume over the previous `window` sessions (data known that day), keep the top N
    for the whole next month. Returns columns date, ticker (one row per eligible day)."""
    base = df["ticker"].str.split("~").str[0]
    d = df[base.isin(cs)].copy()
    d["dv"] = d["close"] * d["volume"]
    # names that never trade $10M in a day can't reach a top-100 list: drop them early
    d = d[d.groupby("ticker")["dv"].transform("max") >= 1e7]
    dv = d.pivot(index="date", columns="ticker", values="dv")
    px = d.pivot(index="date", columns="ticker", values="close")
    med = dv.rolling(window, min_periods=int(window * 0.8)).median()
    hist = px.notna().cumsum()
    naive = dv.index.tz_localize(None) if dv.index.tz is not None else dv.index
    month_end = dv.index.to_series().groupby(naive.to_period("M")).max()
    members = {}
    for per, day in month_end.items():
        ok = (px.loc[day] >= min_price) & (hist.loc[day] >= min_history) & med.loc[day].notna()
        ranked = med.loc[day][ok].sort_values(ascending=False)
        members[per + 1] = list(ranked.index[:top_n])
    rows = []
    for day, nd in zip(dv.index, naive):
        for t in members.get(nd.to_period("M"), []):
            rows.append((day, t))
    return pd.DataFrame(rows, columns=["date", "ticker"])


class MinuteResolver:
    """Decides which barrier was touched first on a day whose high/low touched both,
    from regular-session minute bars (cached on disk, one small request per case)."""

    def __init__(self, client: Optional[MassiveClient], cache: Path = DATA / "minute"):
        self.client, self.cache = client, cache
        self.calls = self.resolved = self.unresolved = 0

    def bars(self, ticker: str, date: str) -> pd.DataFrame:
        sym = ticker.split("~")[0]
        f = self.cache / sym / f"{date}.csv.gz"
        if f.exists():
            return pd.read_csv(f)
        if self.client is None:
            return pd.DataFrame()
        self.calls += 1
        b = pd.DataFrame(self.client.minute_bars(sym, date))
        f.parent.mkdir(parents=True, exist_ok=True)
        b.to_csv(f, index=False, compression="gzip")
        return b

    def __call__(self, ticker: str, date: pd.Timestamp, tp_px: float, sl_px: float, side: int):
        day = pd.Timestamp(date).strftime("%Y-%m-%d")
        b = self.bars(ticker, day)
        if b.empty or "t" not in b:
            self.unresolved += 1
            return None
        ts = pd.to_datetime(b["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        rth = (ts.dt.time >= pd.Timestamp("09:30").time()) & (ts.dt.time < pd.Timestamp("16:00").time())
        b = b[rth.to_numpy()]
        hi, lo = b["h"].to_numpy(), b["l"].to_numpy()
        tp_hit = hi >= tp_px if side > 0 else lo <= tp_px
        sl_hit = lo <= sl_px if side > 0 else hi >= sl_px
        first_tp = np.argmax(tp_hit) if tp_hit.any() else np.inf
        first_sl = np.argmax(sl_hit) if sl_hit.any() else np.inf
        if first_tp == np.inf and first_sl == np.inf:
            self.unresolved += 1
            return None
        self.resolved += 1
        return "tp" if first_tp < first_sl else "sl"  # same minute -> stop (conservative)

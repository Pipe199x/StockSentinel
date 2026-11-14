"""Price data-quality gate, run right after the prices are loaded.

Fatal problems stop the run (the workflow retries, then the backup run tries again): they
would corrupt features, signals or the stateful paper ledger. Statistical anomalies are only
flagged and published, because real crashes look like outliers too. A non-benchmark ticker
that misses the last session is reported as stale so the pipeline can keep it flat that day.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from etl_common.sessions import last_completed_session

PRICE_COLS = ["open", "high", "low", "close"]
SPLIT_RATIOS = (2, 3, 4, 5, 10, 20)   # an unadjusted split shows up as |log r| ~ ln(k)
SPLIT_TOL = 0.03
OUTLIER_FLOOR = 0.25                  # |log return| flagged above max(25%, 10 x rolling median |r|)


@dataclass
class QualityReport:
    fatal: List[str] = field(default_factory=list)
    flags: List[dict] = field(default_factory=list)
    n_flags: int = 0
    stale: List[str] = field(default_factory=list)
    clamped: int = 0
    last_session: Optional[str] = None

    def flag(self, check: str, max_flags: int, **info) -> None:
        self.n_flags += 1
        if len(self.flags) < max_flags:
            self.flags.append({"check": check, **info})

    def as_dict(self) -> dict:
        return asdict(self)


def _session_day(dates: pd.Series) -> pd.Series:
    day = pd.to_datetime(dates)
    if day.dt.tz is not None:
        day = day.dt.tz_convert("UTC").dt.tz_localize(None)
    return day.dt.normalize()


def validate_prices(df: pd.DataFrame, cal: str = "NYSE", now_utc=None, bench: str = "SPY",
                    max_flags: int = 50) -> Tuple[pd.DataFrame, QualityReport]:
    """Returns (prices with minor OHLC inconsistencies clamped, report)."""
    rep = QualityReport()
    need = ["date", "ticker"] + PRICE_COLS + ["volume"]
    missing = [c for c in need if c not in df.columns]
    if df.empty or missing:
        rep.fatal.append("no price rows" if df.empty else f"missing columns: {missing}")
        return df, rep

    df = df.copy()
    day = _session_day(df["date"])
    order = np.lexsort((day.to_numpy(), df["ticker"].to_numpy()))
    df, day = df.iloc[order].reset_index(drop=True), day.iloc[order].reset_index(drop=True)

    dup = int(pd.DataFrame({"t": df["ticker"], "d": day}).duplicated().sum())
    if dup:
        rep.fatal.append(f"{dup} duplicated (ticker, date) rows")

    # Every ticker must reach the last completed session; the benchmark gates trading.
    last_by_ticker = day.groupby(df["ticker"]).max()
    last = last_completed_session(now_utc, cal)
    if last is not None:
        rep.last_session = str(last.date())
        behind = sorted(last_by_ticker[last_by_ticker < last].index)
        if bench in behind:
            rep.fatal.append(f"benchmark {bench} is missing the last completed session {last.date()}")
        if len(behind) > 0.2 * len(last_by_ticker):
            rep.fatal.append(f"{len(behind)} of {len(last_by_ticker)} tickers miss the last session {last.date()}")
        rep.stale = [t for t in behind if t != bench]

    # Prices must be positive and present where they still matter (each ticker's last 5 sessions).
    tail = df.groupby("ticker", group_keys=False).tail(5)
    bad = tail[PRICE_COLS].isna().any(axis=1) | (tail[PRICE_COLS] <= 0).any(axis=1)
    if bad.any():
        rep.fatal.append(f"non-positive or missing OHLC in the last 5 sessions: {sorted(set(tail.loc[bad, 'ticker']))}")
    old_bad = (df[PRICE_COLS] <= 0).any(axis=1) & ~df.index.isin(tail.index)
    for i in np.flatnonzero(old_bad.to_numpy()):
        rep.flag("non_positive_price", max_flags, ticker=df.at[i, "ticker"], date=str(day.iat[i].date()))

    # OHLC consistency: high >= max(open, close) and low <= min(open, close).
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in PRICE_COLS)
    with np.errstate(invalid="ignore"):
        incons = ((h < np.fmax(o, c) * (1 - 1e-6)) | (l > np.fmin(o, c) * (1 + 1e-6))) & np.isfinite(o) & np.isfinite(c)
    if incons.any():
        is_last = (day.to_numpy() == last_by_ticker.reindex(df["ticker"]).to_numpy())
        if (incons & is_last).any() or incons.mean() > 0.001:
            rep.fatal.append(f"inconsistent OHLC on {int(incons.sum())} rows (on a last bar or > 0.1% of rows)")
        else:
            df.loc[incons, "high"] = np.fmax(h, np.fmax(o, c))[incons]
            df.loc[incons, "low"] = np.fmin(l, np.fmin(o, c))[incons]
            rep.clamped = int(incons.sum())
            for i in np.flatnonzero(incons):
                rep.flag("ohlc_clamped", max_flags, ticker=df.at[i, "ticker"], date=str(day.iat[i].date()))

    # Returns: an unadjusted split is fatal; other big moves are flagged.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(df["close"].where(df["close"] > 0)).groupby(df["ticker"]).diff()
    mad = r.abs().groupby(df["ticker"]).transform(lambda s: s.rolling(60, min_periods=20).median().shift(1))
    for i in np.flatnonzero((r.abs() > 0.5).to_numpy()):
        a = abs(float(r.iat[i]))
        if any(abs(a - math.log(k)) < SPLIT_TOL for k in SPLIT_RATIOS):
            rep.fatal.append(f"{df.at[i, 'ticker']} {day.iat[i].date()}: log return {float(r.iat[i]):+.3f} "
                             "looks like an unadjusted split")
    outlier = r.abs() > np.maximum(OUTLIER_FLOOR, 10 * mad.fillna(0.0))
    for i in np.flatnonzero(outlier.to_numpy()):
        rep.flag("outlier_return", max_flags, ticker=df.at[i, "ticker"], date=str(day.iat[i].date()),
                 value=round(float(r.iat[i]), 4))

    zero_vol = (df["volume"].fillna(0) <= 0).to_numpy()
    for i in np.flatnonzero(zero_vol):
        rep.flag("zero_volume", max_flags, ticker=df.at[i, "ticker"], date=str(day.iat[i].date()))

    # Sessions missing inside each ticker's history. A calendar session with no bar for ANY
    # ticker is an unscheduled market closure the calendar library does not know (e.g. the
    # 2025-01-09 national day of mourning): reported once, not as a data gap per ticker.
    try:
        import pandas_market_calendars as mcal
        sessions = pd.DatetimeIndex(mcal.get_calendar(cal).schedule(start_date=day.min().date(),
                                                                    end_date=day.max().date()).index).normalize()
        closures = sessions.difference(pd.DatetimeIndex(day.unique()))
        if len(closures):
            rep.flag("calendar_extra_session", max_flags, value=[str(x.date()) for x in closures])
        for t, d in day.groupby(df["ticker"]):
            expected = sessions[(sessions >= d.min()) & (sessions <= d.max())].difference(closures)
            gap = len(expected.difference(pd.DatetimeIndex(d)))
            if gap:
                rep.flag("missing_sessions", max_flags, ticker=t, value=int(gap))
    except Exception as exc:  # noqa: BLE001 - the calendar is a nicety here, never a reason to fail
        rep.flag("calendar_unavailable", max_flags, value=str(exc)[:120])
    return df, rep

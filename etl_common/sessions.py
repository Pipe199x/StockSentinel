"""NYSE sessions: never use a daily bar from a session that has not closed yet.

A daily bar fetched while the market is open is a partial bar: its high, low, close
and volume are still changing. The 2026-11-13 backup run started at 14:59 UTC
(29 minutes after the open), published signals from that partial bar and marked
every open paper position as checked for the whole day. These helpers let the
pipeline keep only sessions that ended at least `buffer_min` minutes ago. Half days
(e.g. the day after Thanksgiving) come from the exchange calendar.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

DEFAULT_CAL = "NYSE"
BUFFER_MIN = 30


def _now(now_utc=None) -> pd.Timestamp:
    ts = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _schedule(start, end, cal: str = DEFAULT_CAL) -> pd.DataFrame:
    import pandas_market_calendars as mcal
    return mcal.get_calendar(cal).schedule(start_date=start, end_date=end)


def last_completed_session(now_utc=None, cal: str = DEFAULT_CAL,
                           buffer_min: int = BUFFER_MIN) -> Optional[pd.Timestamp]:
    """Last session (naive date at 00:00) whose close + buffer is at or before `now_utc`."""
    now = _now(now_utc)
    sched = _schedule((now - pd.Timedelta(days=20)).date(), now.date(), cal)
    done = sched[sched["market_close"] + pd.Timedelta(minutes=buffer_min) <= now]
    if done.empty:
        return None
    return pd.Timestamp(done.index[-1]).normalize()


def session_open_utc(day, cal: str = DEFAULT_CAL) -> Optional[pd.Timestamp]:
    """Opening time (tz-aware UTC) of the session on `day`; None if the exchange is closed."""
    d = pd.Timestamp(day)
    d = (d.tz_convert("UTC").tz_localize(None) if d.tzinfo is not None else d).normalize()
    sched = _schedule(d.date(), d.date(), cal)
    return None if sched.empty else pd.Timestamp(sched["market_open"].iloc[0])


def next_session(day, cal: str = DEFAULT_CAL) -> Optional[pd.Timestamp]:
    """First session (naive date) strictly after `day`."""
    d = pd.Timestamp(day)
    d = (d.tz_convert("UTC").tz_localize(None) if d.tzinfo is not None else d).normalize()
    sched = _schedule((d + pd.Timedelta(days=1)).date(), (d + pd.Timedelta(days=20)).date(), cal)
    return None if sched.empty else pd.Timestamp(sched.index[0]).normalize()


def drop_incomplete_sessions(df: pd.DataFrame, now_utc=None, cal: str = DEFAULT_CAL,
                             buffer_min: int = BUFFER_MIN, date_col: str = "date") -> pd.DataFrame:
    """Rows whose session has closed (+ buffer). Dates may be naive or tz-aware (session day)."""
    if df.empty:
        return df
    day = pd.to_datetime(df[date_col])
    if day.dt.tz is not None:
        day = day.dt.tz_convert("UTC").dt.tz_localize(None)
    day = day.dt.normalize()
    last = last_completed_session(now_utc, cal, buffer_min)
    if last is None:  # no completed session in the lookback: keep only days strictly before today
        last = _now(now_utc).tz_localize(None).normalize() - pd.Timedelta(days=1)
    return df[(day <= last).to_numpy()]

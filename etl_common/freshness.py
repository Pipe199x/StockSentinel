"""Workflow guard: skip a scheduled run when the published data already covers the last
completed session. Standard library only, so it runs before any dependency is installed.

    python3 -m etl_common.freshness docs/data/manifest.json   # prints skip=true|false

The session end is approximated as 21:30 UTC on weekdays: the NYSE close in winter
(21:00 UTC) plus the 30-minute buffer the ETL uses, which is also late enough in summer
(close 20:00 UTC). Holidays are unknown here: on a holiday the guard expects a session
that never happened and lets one harmless extra run through (the ETL itself drops
incomplete sessions with the exchange calendar, see etl_common/sessions.py).
"""
import json
import sys
from datetime import date, datetime, timedelta, timezone

SESSION_END_UTC = (21, 30)


def expected_last_session(now_utc: datetime) -> date:
    d = now_utc.date()
    end = datetime(d.year, d.month, d.day, *SESSION_END_UTC, tzinfo=timezone.utc)
    if now_utc < end:
        d -= timedelta(days=1)
    while d.weekday() >= 5:  # Saturday, Sunday
        d -= timedelta(days=1)
    return d


def should_skip(manifest_path: str, now_utc: datetime) -> bool:
    """True when manifest.stocks.as_of already covers the expected last session."""
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            as_of = date.fromisoformat(json.load(fh)["stocks"]["as_of"])
    except Exception:  # unreadable or missing manifest: run the ETL
        return False
    return as_of >= expected_last_session(now_utc)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/data/manifest.json"
    now = datetime.now(timezone.utc)
    print(f"skip={'true' if should_skip(path, now) else 'false'}")

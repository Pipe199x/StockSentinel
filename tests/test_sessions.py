"""Complete-session guard: a bar from a session that has not closed must never be used."""
import json
from datetime import date, datetime, timezone

import pandas as pd

from etl_common.freshness import expected_last_session, should_skip
from etl_common.sessions import (drop_incomplete_sessions, last_completed_session, next_session,
                                 session_open_utc)

D = pd.Timestamp


def test_partial_session_is_not_completed():
    # 2026-11-13 incident: the backup run started at 14:59 UTC, 29 minutes after the open
    assert last_completed_session("2026-11-13T14:59:00Z") == D("2026-11-12")


def test_close_plus_buffer_boundary_est():
    # 13 Nov 2026 is EST: close 21:00 UTC, +30 min buffer
    assert last_completed_session("2026-11-13T21:29:00Z") == D("2026-11-12")
    assert last_completed_session("2026-11-13T21:30:00Z") == D("2026-11-13")


def test_thanksgiving_and_half_day():
    # Thanksgiving (26 Nov) is closed; 27 Nov is a half day closing at 18:00 UTC
    assert last_completed_session("2026-11-27T03:00:00Z") == D("2026-11-25")
    assert last_completed_session("2026-11-27T18:29:00Z") == D("2026-11-25")
    assert last_completed_session("2026-11-27T18:30:00Z") == D("2026-11-27")


def test_weekend_points_to_friday():
    assert last_completed_session("2026-11-15T12:00:00Z") == D("2026-11-13")


def test_drop_incomplete_sessions_tz_aware_and_naive():
    days = pd.to_datetime(["2026-11-11", "2026-11-12", "2026-11-13"])
    for dates in (days.tz_localize("UTC"), days):
        df = pd.DataFrame({"date": dates, "ticker": "AAA", "close": [1.0, 2.0, 3.0]})
        out = drop_incomplete_sessions(df, "2026-11-13T14:59:00Z")
        assert list(out["close"]) == [1.0, 2.0]
        assert len(drop_incomplete_sessions(df, "2026-11-14T02:00:00Z")) == 3


def test_session_open_and_next_session():
    assert session_open_utc("2026-11-13") == D("2026-11-13 14:30", tz="UTC")
    assert session_open_utc("2026-11-26") is None          # Thanksgiving
    assert next_session("2026-11-25") == D("2026-11-27")
    assert next_session(D("2026-11-13", tz="UTC")) == D("2026-11-16")


def _utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_expected_last_session_stdlib_guard():
    assert expected_last_session(_utc("2026-11-14T01:30:00")) == date(2026, 11, 13)   # primary run
    assert expected_last_session(_utc("2026-11-13T14:59:00")) == date(2026, 11, 12)   # mid-session
    assert expected_last_session(_utc("2026-11-15T01:30:00")) == date(2026, 11, 13)   # Sunday
    assert expected_last_session(_utc("2026-11-17T01:30:00")) == date(2026, 11, 16)   # Tuesday


def test_should_skip(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"stocks": {"as_of": "2026-11-13"}}), encoding="utf-8")
    assert should_skip(str(m), _utc("2026-11-14T04:00:00"))        # backup after a good primary
    assert should_skip(str(m), _utc("2026-11-16T01:30:00"))        # weekend rerun
    assert not should_skip(str(m), _utc("2026-11-17T01:30:00"))    # Monday's session not published yet
    assert not should_skip(str(tmp_path / "missing.json"), _utc("2026-11-14T04:00:00"))

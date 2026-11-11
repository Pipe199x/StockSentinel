import numpy as np
import pandas as pd

import research.replay.paper_replay as rp
from stocks_etl.meta_model import FEATURES


class _Spy:
    """Stand-in model that records what it was trained on."""
    seen = []

    def fit(self, X, y):
        _Spy.seen.append(X.index)
        return self

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


def _events(n=600):
    start = pd.bdate_range("2023-01-02", periods=n)
    ev = pd.DataFrame({c: np.random.default_rng(0).normal(size=n) for c in FEATURES})
    ev["start"] = start
    ev["end"] = start + pd.offsets.BDay(10)
    ev["label"] = np.arange(n) % 2
    return ev


def test_monthly_retrain_only_sees_resolved_labels(monkeypatch):
    monkeypatch.setattr(rp, "_model", lambda seed: _Spy())
    _Spy.seen.clear()
    ev = _events()
    cand = ev.iloc[-60:].copy()
    cand["day"] = cand["start"]
    prob = rp.walk_forward_probs(ev, cand.reset_index(drop=True), {}, embargo_days=5)
    months = cand["day"].dt.to_period("M").unique()
    assert len(_Spy.seen) == len(months)
    for m, idx in zip(months, _Spy.seen):
        cutoff = m.start_time - pd.offsets.BDay(5)
        assert (ev.loc[idx, "end"] < cutoff).all()
    assert np.allclose(prob, 0.6)


def test_replay_ledger_never_sees_future_prices(monkeypatch):
    days = pd.bdate_range("2026-01-05", periods=6, tz="UTC")
    prices = pd.DataFrame([{"date": d, "ticker": "AAA", "open": 100, "high": 101, "low": 99, "close": 100}
                           for d in days])
    seen = []
    real = rp.update_ledger

    def spy(ledger, px, sig, cost):
        seen.append(pd.to_datetime(px["date"]).max())
        return real(ledger, px, sig, cost)

    monkeypatch.setattr(rp, "update_ledger", spy)
    sig = pd.DataFrame([{"ticker": "AAA", "as_of": str(days[1].date()), "side": 1, "kelly_size": 0.05,
                         "p_success": 0.6, "tp": 110.0, "sl": 90.0, "t_max_date": str(days[4].date())}])
    led = rp.replay_ledger(prices, sig, days[0].tz_localize(None).normalize(), 5)
    assert [s.date() for s in seen] == [d.date() for d in days]
    r = led.iloc[0]
    assert r.status == "closed" and r.exit_reason == "time" and r.entry_date == str(days[2].date())

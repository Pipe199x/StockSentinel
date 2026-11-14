"""Simulated paper trading (Gate 0): the published signals, executed as a trader would.

No broker, no keys. Every run:
  1. Positions waiting for entry ('pending') fill at the OPEN of the first session after
     their signal date, keeping the published take-profit, stop-loss and exit date.
  2. Open positions are walked session by session with the same rules as the backtest
     (labeling.triple_barrier_live): a gap through a barrier fills at the open, an intraday
     touch fills at the barrier, both barriers on the same day count as the stop, and the
     time limit closes at that session's close. The walk always restarts at the ENTRY day,
     so the result is a pure function of the stored entry/barriers and the complete bars:
     a session first seen as a partial bar is re-checked once its full bar exists.
     Closed rows are re-verified the same way; any change is reported, never silent.
  3. Today's signals with a non-zero size become new 'pending' rows (one position per
     ticker at a time). Rows are keyed by ticker + signal date, so re-runs never duplicate.

The ledger is the source of truth (docs/data/history/paper_trades.csv); the summary JSON
compares live results with the backtest range: Gate 0 passes only on a large enough sample.
"""
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stocks_etl.validation import sharpe

COLUMNS = ["ticker", "signal_date", "side", "size", "p_success", "tp", "sl", "exit_by", "status",
           "entry_date", "entry_price", "last_checked", "exit_date", "exit_price", "exit_reason",
           "ret_gross", "ret_net"]
OUTCOME = ["exit_date", "exit_price", "exit_reason", "ret_gross", "ret_net"]
MIN_TRADES = 30  # below this the comparison is noise (see 1 Nov / 4 Nov videos)


def load_ledger(path: str) -> pd.DataFrame:
    if Path(path).exists():
        led = pd.read_csv(path, dtype={"ticker": str, "status": str, "exit_reason": str})
        for c in COLUMNS:
            if c not in led:
                led[c] = np.nan
        return led[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def _bars(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    p = prices.copy()
    p["day"] = pd.to_datetime(p["date"]).dt.tz_localize(None).dt.normalize()
    p = p.dropna(subset=["close"])
    return {t: g.set_index("day").sort_index()[["open", "high", "low", "close"]] for t, g in p.groupby("ticker")}


def _walk(row: dict, bars: pd.DataFrame, cost: float) -> dict:
    """Walk one position from its ENTRY day through every complete session in `bars`.

    A pending row first fills at the open of the session after its signal date. From there
    the outcome is recomputed from scratch on every run (never resumed from `last_checked`),
    so it only depends on entry price, barriers, exit date and complete bars.
    """
    s = int(row["side"])
    tp, sl = float(row["tp"]), float(row["sl"])
    exit_by = pd.Timestamp(row["exit_by"])
    if row["status"] == "pending":
        after = bars[bars.index > pd.Timestamp(row["signal_date"])]
        if after.empty:
            return row
        day = after.index[0]
        o = after["open"].iloc[0]
        entry = float(o) if np.isfinite(o) else float(after["close"].iloc[0])
        row.update(status="open", entry_date=str(day.date()), entry_price=entry)
    row = dict(row, status="open", **{k: np.nan for k in OUTCOME})
    start = pd.Timestamp(row["entry_date"])
    first = True
    entry = float(row["entry_price"])
    for day, b in bars[bars.index >= start].iterrows():
        o, h, l, c = (float(b[k]) for k in ("open", "high", "low", "close"))
        px, reason = None, None
        if first and (s * (entry - tp) >= 0 or s * (entry - sl) <= 0):
            px, reason = entry, "entry_gap"  # the open already crossed a barrier: the bracket fills flat
        elif not first and np.isfinite(o):
            if s * (o - tp) >= 0:          # opened beyond take-profit
                px, reason = o, "tp"
            elif s * (o - sl) <= 0:        # opened beyond the stop: worse than the stop
                px, reason = o, "sl"
        if px is None:
            hit_tp = (h >= tp) if s > 0 else (l <= tp)
            hit_sl = (l <= sl) if s > 0 else (h >= sl)
            if hit_sl:                     # stop first when both touched (conservative)
                px, reason = sl, "sl"
            elif hit_tp:
                px, reason = tp, "tp"
        if px is None and day >= exit_by:
            px, reason = c, "time"
        first = False
        row["last_checked"] = str(day.date())
        if px is not None:
            g = s * (px / entry - 1)
            row.update(status="closed", exit_date=str(day.date()), exit_price=px, exit_reason=reason,
                       ret_gross=g, ret_net=g - cost)
            break
    return row


def _same_outcome(a: dict, b: dict) -> bool:
    if str(a.get("exit_date")) != str(b.get("exit_date")) or str(a.get("exit_reason")) != str(b.get("exit_reason")):
        return False
    return all(np.isclose(float(a[k]), float(b[k]), rtol=1e-9, atol=1e-12, equal_nan=True)
               for k in ("exit_price", "ret_net"))


def _verify_closed(row: dict, bars: pd.DataFrame, cost: float, revisions: Optional[List[dict]]) -> dict:
    """Re-walk a closed row; replace it only if complete bars now give another outcome."""
    new = _walk(dict(row), bars, cost)
    if new["status"] != "closed" or _same_outcome(row, new):
        return row  # unchanged, or the bars no longer reach the exit (e.g. a run with an earlier --now)
    if revisions is not None:
        revisions.append({"ticker": row["ticker"], "signal_date": str(row["signal_date"]),
                          "old": {k: row.get(k) for k in OUTCOME}, "new": {k: new.get(k) for k in OUTCOME}})
    return new


def update_ledger(ledger: pd.DataFrame, prices: pd.DataFrame, signals: pd.DataFrame, cost_bps: float,
                  revisions: Optional[List[dict]] = None) -> pd.DataFrame:
    bars = _bars(prices)
    cost = 2 * float(cost_bps) / 1e4
    rows = []
    for row in ledger.to_dict("records"):
        if row["ticker"] in bars:
            if row["status"] in ("pending", "open"):
                row = _walk(row, bars[row["ticker"]], cost)
            elif row["status"] == "closed":
                row = _verify_closed(row, bars[row["ticker"]], cost, revisions)
        rows.append(row)
    seen = {(r["ticker"], str(r["signal_date"])) for r in rows}
    busy = {r["ticker"] for r in rows if r["status"] in ("pending", "open")}
    for sig in signals.to_dict("records"):
        size = sig.get("kelly_size") or 0.0
        if not size or size <= 0 or not sig.get("side") or sig.get("tp") is None:
            continue
        key = (sig["ticker"], str(sig["as_of"]))
        if key in seen or sig["ticker"] in busy:
            continue
        rows.append({"ticker": sig["ticker"], "signal_date": str(sig["as_of"]), "side": int(sig["side"]),
                     "size": float(size), "p_success": sig.get("p_success"), "tp": float(sig["tp"]),
                     "sl": float(sig["sl"]), "exit_by": sig["t_max_date"], "status": "pending"})
        busy.add(sig["ticker"])
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def equity_curve(ledger: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Compounding equity (start = 1): realized P&L at each exit, open positions marked to the close."""
    trades = ledger[ledger["status"].isin(["open", "closed"])]
    if trades.empty:
        return pd.DataFrame(columns=["date", "equity"])
    bars = _bars(prices)
    start = pd.Timestamp(trades["entry_date"].min())
    days = sorted({d for t in trades["ticker"].unique() if t in bars for d in bars[t].index if d >= start})
    closed = trades[trades["status"] == "closed"].sort_values("exit_date")
    out = []
    for day in days:
        done = closed[pd.to_datetime(closed["exit_date"]) <= day]
        realized = float(np.prod(1 + done["size"].astype(float) * done["ret_net"].astype(float))) if len(done) else 1.0
        live = trades[(pd.to_datetime(trades["entry_date"]) <= day) &
                      ((trades["status"] == "open") | (pd.to_datetime(trades["exit_date"]) > day))]
        unreal = 0.0
        for r in live.itertuples():
            px = bars[r.ticker]["close"].get(day, np.nan)
            if np.isfinite(px):
                unreal += float(r.size) * int(r.side) * (px / float(r.entry_price) - 1)
        out.append({"date": str(day.date()), "equity": realized * (1 + unreal)})
    return pd.DataFrame(out)


def summarize(ledger: pd.DataFrame, prices: pd.DataFrame, report: Dict, as_of: str) -> Dict:
    closed = ledger[ledger["status"] == "closed"]
    ret = closed["ret_net"].astype(float) if len(closed) else pd.Series(dtype=float)
    pnl = (closed["size"].astype(float) * ret) if len(closed) else pd.Series(dtype=float)
    curve = equity_curve(ledger, prices)
    live_sr = sharpe(pnl) if len(pnl) > 1 else None
    cpcv = [x for x in (report.get("cpcv_path_sr") or []) if x is not None and math.isfinite(x)]
    bt = report.get("meta", {})
    if len(closed) < MIN_TRADES:
        status, why = "collecting", f"{len(closed)} of {MIN_TRADES} closed trades needed before judging"
    elif live_sr is not None and cpcv and live_sr >= min(cpcv) and ret.mean() > 0:
        status, why = "pass", "live Sharpe per trade is inside the backtest range and the edge is positive"
    else:
        status, why = "fail", "live results fall below the backtest range or the edge is not positive"

    def recs(df: pd.DataFrame, n: int = None):
        d = df.where(pd.notna(df), None)
        return d.tail(n).to_dict("records") if n else d.to_dict("records")

    return {
        "as_of": as_of,
        "started": str(ledger["signal_date"].min()) if len(ledger) else None,
        "counts": {s: int((ledger["status"] == s).sum()) for s in ("pending", "open", "closed")},
        "win_rate": float((ret > 0).mean()) if len(ret) else None,
        "expected_value": float(ret.mean()) if len(ret) else None,
        "sr_per_trade": live_sr,
        "total_return": float(curve["equity"].iloc[-1] - 1) if len(curve) else 0.0,
        "equity_curve": curve.to_dict("records"),
        "positions": recs(ledger[ledger["status"].isin(["pending", "open"])]),
        "recent_closed": recs(closed.sort_values("exit_date"), 25),
        "gate0": {"status": status, "reason": why, "min_trades": MIN_TRADES,
                  "live_sr_per_trade": live_sr,
                  "backtest_sr_per_trade": bt.get("sr_per_trade"),
                  "backtest_expected_value": bt.get("expected_value"),
                  "backtest_cpcv_range": [min(cpcv), max(cpcv)] if cpcv else None},
    }


def run_paper(prices: pd.DataFrame, signals: pd.DataFrame, report: Dict, cfg: dict,
              ledger_path: str) -> Tuple[pd.DataFrame, Dict]:
    revisions: List[dict] = []
    led = update_ledger(load_ledger(ledger_path), prices, signals, cfg.get("cost_bps", 5), revisions=revisions)
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(ledger_path, index=False)
    for r in revisions:
        print(f"[PAPER] revised closed row {r['ticker']} {r['signal_date']}: {r['old']} -> {r['new']}")
    as_of = str(pd.to_datetime(prices["date"]).max().date())
    summary = summarize(led, prices, report, as_of)
    summary["ledger_integrity"] = {"revised_rows": revisions}
    return led, summary

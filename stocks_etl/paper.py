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
from scipy.stats import kurtosis, skew

from etl_common.sessions import next_session, session_open_utc
from stocks_etl.validation import sharpe

# Appended columns keep older ledgers readable: published_utc (when the signal went out),
# config_hash / strategy_label (which configuration produced it).
COLUMNS = ["ticker", "signal_date", "side", "size", "p_success", "tp", "sl", "exit_by", "status",
           "entry_date", "entry_price", "last_checked", "exit_date", "exit_price", "exit_reason",
           "ret_gross", "ret_net", "published_utc", "config_hash", "strategy_label", "div_cash"]
OUTCOME = ["exit_date", "exit_price", "exit_reason", "ret_gross", "ret_net", "div_cash"]
MIN_TRADES = 30  # below this the comparison is noise (see 1 Nov / 4 Nov videos)
Z95_ONE_SIDED = 1.6449

# Rows written before the ledger was tagged. The 2026-11-11 signals came from the V4 run of
# 2026-11-12 03:14 UTC (trial f71882e35e28); from 2026-11-12 on, V4 + news (c3f2b4fc9bfd).
LEGACY_TAGS = (("2026-11-11", "f71882e35e28", "V4"), ("9999-12-31", "c3f2b4fc9bfd", "V4+news"))


def _tag_legacy(led: pd.DataFrame) -> pd.DataFrame:
    """Idempotent: only rows without a config_hash are tagged, by signal date."""
    untagged = led["config_hash"].isna() | (led["config_hash"].astype(str).str.strip() == "")
    if not untagged.any():
        return led
    led = led.copy()
    led["config_hash"] = led["config_hash"].astype(object)
    led["strategy_label"] = led["strategy_label"].astype(object)
    for i in np.flatnonzero(untagged.to_numpy()):
        sd = str(led["signal_date"].iat[i])
        h, label = next((h, lab) for upto, h, lab in LEGACY_TAGS if sd <= upto)
        led.iat[i, led.columns.get_loc("config_hash")] = h
        led.iat[i, led.columns.get_loc("strategy_label")] = label
    return led


def load_ledger(path: str) -> pd.DataFrame:
    if Path(path).exists():
        led = pd.read_csv(path, dtype={"ticker": str, "status": str, "exit_reason": str, "config_hash": str,
                                       "strategy_label": str, "published_utc": str})
        for c in COLUMNS:
            if c not in led:
                led[c] = np.nan
        return _tag_legacy(led[COLUMNS])
    return pd.DataFrame(columns=COLUMNS)


def _published_after_open(signal_date, published_utc, cal: str) -> bool:
    """True when a signal was published after the open of the session it should fill at:
    a real trader could not have bought that open, so the paper bot does not either."""
    if not isinstance(published_utc, str) or not published_utc:
        return False
    nxt = next_session(signal_date, cal)
    opening = session_open_utc(nxt, cal) if nxt is not None else None
    return opening is not None and pd.Timestamp(published_utc) > opening


def _bars(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    p = prices.copy()
    p["day"] = pd.to_datetime(p["date"]).dt.tz_localize(None).dt.normalize()
    p = p.dropna(subset=["close"])
    if "dividends" not in p:
        p["dividends"] = 0.0
    p["dividends"] = p["dividends"].fillna(0.0).astype(float)
    cols = ["open", "high", "low", "close", "dividends"]
    return {t: g.set_index("day").sort_index()[cols] for t, g in p.groupby("ticker")}


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
    div = 0.0   # cash dividends with ex-date in (entry day, today]: longs receive, shorts pay
    for day, b in bars[bars.index >= start].iterrows():
        o, h, l, c = (float(b[k]) for k in ("open", "high", "low", "close"))
        if not first:
            div += float(b["dividends"])
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
            g = s * ((px + div) / entry - 1)
            row.update(status="closed", exit_date=str(day.date()), exit_price=px, exit_reason=reason,
                       ret_gross=g, ret_net=g - cost, div_cash=div)
            break
    return row


SPLIT_TOL = 0.02


def rebase_for_splits(ledger: pd.DataFrame, prices: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    """Keep stored prices on the scale of the split-adjusted bars.

    When a split happens during (or after) a position, Massive re-adjusts the whole history, so
    the ledger's entry, TP, SL, exit and dividend amounts are left on the old scale and every
    barrier would look crossed. If the entry day's open now differs from the stored entry price
    by more than 2% (a split ratio, not a data revision), all of the row's price fields are
    rescaled by that ratio; returns are ratios and do not change. Pending rows have no fill to
    compare with and are left alone (a split between signal and fill is not handled).
    """
    bars = _bars(prices)
    out = ledger.copy()
    rebased: List[dict] = []
    for i in range(len(out)):
        r = out.iloc[i]
        if r["status"] not in ("open", "closed") or r["ticker"] not in bars or pd.isna(r["entry_date"]):
            continue
        b, d = bars[r["ticker"]], pd.Timestamp(r["entry_date"])
        if d not in b.index or pd.isna(r["entry_price"]) or float(r["entry_price"]) <= 0:
            continue
        o = float(b.at[d, "open"]) if np.isfinite(b.at[d, "open"]) else float(b.at[d, "close"])
        k = o / float(r["entry_price"])
        if not np.isfinite(k) or abs(k - 1) <= SPLIT_TOL:
            continue
        for c in ("entry_price", "tp", "sl", "exit_price", "div_cash"):
            if pd.notna(out[c].iat[i]):
                out.iat[i, out.columns.get_loc(c)] = float(out[c].iat[i]) * k
        rebased.append({"ticker": r["ticker"], "signal_date": str(r["signal_date"]), "factor": k})
    return out, rebased


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
                  revisions: Optional[List[dict]] = None, published_utc: Optional[str] = None,
                  config_hash: Optional[str] = None, strategy_label: Optional[str] = None,
                  cal: str = "NYSE") -> pd.DataFrame:
    bars = _bars(prices)
    cost = 2 * float(cost_bps) / 1e4
    rows = []
    for row in ledger.to_dict("records"):
        if row["status"] == "pending" and _published_after_open(row["signal_date"], row.get("published_utc"), cal):
            row.update(status="missed", exit_reason="late_publish")
        elif row["ticker"] in bars:
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
        late = _published_after_open(str(sig["as_of"]), published_utc, cal)
        rows.append({"ticker": sig["ticker"], "signal_date": str(sig["as_of"]), "side": int(sig["side"]),
                     "size": float(size), "p_success": sig.get("p_success"), "tp": float(sig["tp"]),
                     "sl": float(sig["sl"]), "exit_by": sig["t_max_date"],
                     "status": "missed" if late else "pending", "exit_reason": "late_publish" if late else np.nan,
                     "published_utc": published_utc, "config_hash": config_hash, "strategy_label": strategy_label})
        if not late:
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
            b = bars[r.ticker]
            px = b["close"].get(day, np.nan)
            if np.isfinite(px):
                d = float(b["dividends"][(b.index > pd.Timestamp(r.entry_date)) & (b.index <= day)].sum())
                unreal += float(r.size) * int(r.side) * ((px + d) / float(r.entry_price) - 1)
        out.append({"date": str(day.date()), "equity": realized * (1 + unreal)})
    return pd.DataFrame(out)


def sr_standard_error(pnl: pd.Series, sr: float) -> float:
    """Standard error of a per-trade Sharpe with skew and (Pearson) kurtosis (Mertens / Lo)."""
    n = len(pnl)
    if n < 3:
        return float("nan")
    g3 = float(skew(pnl)) if n > 2 else 0.0
    g4 = float(kurtosis(pnl, fisher=False)) if n > 3 else 3.0
    return float(math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12) / (n - 1)))


def gate0(pnl: pd.Series, ret: pd.Series, report: Dict, min_trades: int = MIN_TRADES) -> Dict:
    """Is the live sample consistent with the backtest? (Not proof of an edge: Gate 1 is.)

    With ~30 trades the standard error of the per-trade Sharpe is ~0.18, far wider than the
    CPCV range, so the gate only FAILS when live results are significantly below the backtest
    (one-sided 95% upper bound under the CPCV median); it PASSES when they are not and the live
    edge is positive; otherwise it is INCONCLUSIVE. min_detectable_gap states the low power.
    """
    cpcv = [x for x in (report.get("cpcv_path_sr") or []) if x is not None and math.isfinite(x)]
    bt = report.get("meta", {})
    n = len(pnl)
    live_sr = sharpe(pnl) if n > 1 else None
    out = {"min_trades": min_trades, "n_closed": int(n), "live_sr_per_trade": live_sr,
           "backtest_sr_per_trade": bt.get("sr_per_trade"), "backtest_expected_value": bt.get("expected_value"),
           "backtest_cpcv_range": [min(cpcv), max(cpcv)] if cpcv else None,
           "backtest_cpcv_median": float(np.median(cpcv)) if cpcv else None,
           "sr_ci95": None, "min_detectable_gap": None}
    if n < min_trades:
        return {**out, "status": "collecting", "reason": f"{n} of {min_trades} closed trades needed before judging"}
    se = sr_standard_error(pnl, live_sr)
    out.update(sr_ci95=[live_sr - 1.96 * se, live_sr + 1.96 * se], min_detectable_gap=Z95_ONE_SIDED * se)
    if cpcv and live_sr + Z95_ONE_SIDED * se < out["backtest_cpcv_median"]:
        return {**out, "status": "fail",
                "reason": "live Sharpe per trade is significantly below the backtest (CPCV median)"}
    if ret.mean() > 0:
        return {**out, "status": "pass",
                "reason": "live results are consistent with the backtest and the live edge is positive"}
    return {**out, "status": "inconclusive",
            "reason": "live results are not significantly below the backtest, but the live edge is not positive"}


def summarize(ledger: pd.DataFrame, prices: pd.DataFrame, report: Dict, as_of: str) -> Dict:
    """Gate 0 and the trade statistics use only trades of the CURRENT configuration
    (report['config_hash']); the equity curve and positions cover the whole account."""
    current = report.get("config_hash")
    mine = ledger[ledger["config_hash"] == current] if current and "config_hash" in ledger else ledger
    closed = mine[mine["status"] == "closed"]
    ret = closed["ret_net"].astype(float) if len(closed) else pd.Series(dtype=float)
    pnl = (closed["size"].astype(float) * ret) if len(closed) else pd.Series(dtype=float)
    curve = equity_curve(ledger, prices)

    def recs(df: pd.DataFrame, n: int = None):
        d = df.astype(object).where(pd.notna(df), None)
        return d.tail(n).to_dict("records") if n else d.to_dict("records")

    by_config, switches = [], []
    if "config_hash" in ledger and len(ledger):
        tagged = ledger.assign(config_hash=ledger["config_hash"].fillna("untagged"))
        for h, g in tagged.groupby("config_hash", sort=False):
            c = g[g["status"] == "closed"]["ret_net"].astype(float)
            label = g["strategy_label"].dropna()
            by_config.append({"config_hash": h, "strategy_label": label.iloc[0] if len(label) else None,
                              "first_signal": str(g["signal_date"].min()), "closed": int(len(c)),
                              "open": int(g["status"].isin(["pending", "open"]).sum()),
                              "missed": int((g["status"] == "missed").sum()),
                              "expected_value": float(c.mean()) if len(c) else None,
                              "win_rate": float((c > 0).mean()) if len(c) else None})
        by_config.sort(key=lambda r: r["first_signal"])
        switches = [{"from_signal_date": r["first_signal"], "config_hash": r["config_hash"],
                     "strategy_label": r["strategy_label"]} for r in by_config]

    return {
        "as_of": as_of,
        "started": str(ledger["signal_date"].min()) if len(ledger) else None,
        "config_hash": current,
        "counts": {s: int((ledger["status"] == s).sum()) for s in ("pending", "open", "closed", "missed")},
        "current_closed": int(len(closed)),
        "win_rate": float((ret > 0).mean()) if len(ret) else None,
        "expected_value": float(ret.mean()) if len(ret) else None,
        "sr_per_trade": sharpe(pnl) if len(pnl) > 1 else None,
        "total_return": float(curve["equity"].iloc[-1] - 1) if len(curve) else 0.0,
        "equity_curve": curve.to_dict("records"),
        "positions": recs(ledger[ledger["status"].isin(["pending", "open"])]),
        "recent_closed": recs(ledger[ledger["status"] == "closed"].sort_values("exit_date"), 25),
        "by_config": by_config,
        "config_switches": switches,
        "gate0": gate0(pnl, ret, report),
    }


def run_paper(prices: pd.DataFrame, signals: pd.DataFrame, report: Dict, cfg: dict,
              ledger_path: str, now=None, strategy_label: Optional[str] = None,
              cal: str = "NYSE") -> Tuple[pd.DataFrame, Dict]:
    revisions: List[dict] = []
    published = (pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")).isoformat()
    led, rebased = rebase_for_splits(load_ledger(ledger_path), prices)
    for r in rebased:
        print(f"[PAPER] split rebase {r['ticker']} {r['signal_date']}: prices x {r['factor']:.6f}")
    led = update_ledger(led, prices, signals, cfg.get("cost_bps", 5), revisions=revisions,
                        published_utc=published, config_hash=report.get("config_hash"),
                        strategy_label=strategy_label, cal=cal)
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(ledger_path, index=False)
    for r in revisions:
        print(f"[PAPER] revised closed row {r['ticker']} {r['signal_date']}: {r['old']} -> {r['new']}")
    as_of = str(pd.to_datetime(prices["date"]).max().date())
    summary = summarize(led, prices, report, as_of)
    summary["ledger_integrity"] = {"revised_rows": revisions, "rebased_for_splits": rebased}
    return led, summary

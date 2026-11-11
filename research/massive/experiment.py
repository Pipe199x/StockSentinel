"""Re-validate the strategy on Massive data: does the edge survive the two fixes?

    python -m research.massive.experiment [--top-n 100] [--no-minutes]

Four scenarios, same model and parameters as the live pipeline:
  S0 survivors_close     the 25 live tickers, barriers on daily closes (= today's method;
                         sanity check that Massive data reproduces the published numbers)
  S1 survivors_intraday  same tickers, barriers on intraday highs/lows (+ minute tie-break)
  S2 pit_close           point-in-time top-N incl. delisted names, close barriers
  S3 pit_intraday        point-in-time + intraday barriers: the honest answer

Every scenario is logged as a trial in reports/massive/trials_log.csv, which starts as a
copy of the live trials log, so the Deflated Sharpe pays for all of these attempts.
Outputs reports/massive/results.json and results.md. Raw Massive data stays in data/.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from research.massive.client import MassiveClient, load_api_key
from research.massive.universe import (ALWAYS, DATA, MinuteResolver, common_stocks, load_grouped, pit_universe,
                                       split_reused_tickers)
from stocks_etl.features import make_features
from stocks_etl.meta_model import make_signals
from stocks_etl.stocks_etl import universe_symbols

OUT = Path("reports/massive")
KEYS = ["n_trades", "years", "win_rate", "expected_value", "sr_per_trade", "sr_annualized", "dsr",
        "max_losing_streak", "min_trades_for_edge"]


def features(df, qcfg, eligible=None):
    """Features are the same for close and intraday barriers: compute once per universe."""
    print(f"[features] {df['ticker'].nunique()} tickers (GARCH + HMM per ticker, this is the slow part)")
    feats, meta = make_features(df, qcfg)
    if eligible is not None:
        e = eligible.assign(eligible=True)
        feats = feats.merge(e, on=["date", "ticker"], how="left")
        feats["eligible"] = feats["eligible"].fillna(False).astype(bool)
        feats.loc[feats["ticker"].isin(ALWAYS), "eligible"] = True
    return feats, meta, int(df["ticker"].nunique())


def run(name, fm, qcfg, cal, trials, resolver=None, barrier_mode="close", universe_tag=""):
    feats, meta, n_tickers = fm
    print(f"\n=== {name}: {n_tickers} tickers, barriers on {barrier_mode} ===")
    cfg = dict(qcfg, barrier_mode=barrier_mode, data_source="massive", universe=universe_tag)
    _, rep = make_signals(feats, meta, cfg, cal_name=cal, trials_path=str(trials), resolver=resolver)
    m = rep["meta"]
    out = {k: m.get(k) for k in KEYS}
    out.update({"n_events": rep["n_events"], "n_trials": rep["n_trials"],
                "edge_is_significant": rep["edge_is_significant"],
                "primary_sr_annualized": rep["primary"].get("sr_annualized"),
                "cpcv_path_sr": rep["cpcv_path_sr"], "tickers": n_tickers})
    print({k: out[k] for k in ("n_trades", "expected_value", "sr_annualized", "dsr")})
    return out


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.3f}%" if pct else (f"{v:,.3f}" if isinstance(v, float) else f"{v:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="stocks_etl/config.yaml")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--no-minutes", action="store_true", help="skip minute tie-breaks (no API calls)")
    ap.add_argument("--only", nargs="*", help="run a subset, e.g. --only S0 S1")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    qcfg = dict(cfg.get("quant") or {}, mc_paths=1000)
    cal = cfg.get("calendar")
    OUT.mkdir(parents=True, exist_ok=True)
    trials = OUT / "trials_log.csv"
    live_trials = Path(cfg.get("publish", {}).get("docs_dir", "docs/data")) / "history" / "trials_log.csv"
    if not trials.exists() and live_trials.exists():
        shutil.copy(live_trials, trials)

    resolver = None if a.no_minutes else MinuteResolver(MassiveClient(load_api_key()))
    want = set(a.only or ["S0", "S1", "S2", "S3"])
    results = {}

    if want & {"S0", "S1"}:
        surv = features(load_grouped(tickers=universe_symbols(cfg)), qcfg)
        if "S0" in want:
            results["S0 survivors_close"] = run("S0", surv, qcfg, cal, trials, universe_tag="live25")
        if "S1" in want:
            results["S1 survivors_intraday"] = run("S1", surv, qcfg, cal, trials, resolver=resolver,
                                                   barrier_mode="intraday", universe_tag="live25")
    if want & {"S2", "S3"}:
        allp = split_reused_tickers(load_grouped())
        members = pit_universe(allp, common_stocks(), top_n=a.top_n)
        keep = set(members["ticker"]) | set(ALWAYS)
        pit = allp[allp["ticker"].isin(keep)].copy()
        print(f"[pit] {members['ticker'].nunique()} distinct stocks were ever in the monthly top {a.top_n}")
        tag = f"pit_top{a.top_n}"
        pit = features(pit, qcfg, eligible=members)
        if "S2" in want:
            results["S2 pit_close"] = run("S2", pit, qcfg, cal, trials, universe_tag=tag)
        if "S3" in want:
            results["S3 pit_intraday"] = run("S3", pit, qcfg, cal, trials, resolver=resolver,
                                             barrier_mode="intraday", universe_tag=tag)

    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(), "top_n": a.top_n,
            "minute_calls": getattr(resolver, "calls", 0), "minute_resolved": getattr(resolver, "resolved", 0),
            "minute_unresolved": getattr(resolver, "unresolved", 0)}
    prev = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    prev_res = prev.get("results", {})
    prev_res.update(results)
    (OUT / "results.json").write_text(json.dumps({"run": meta, "results": prev_res}, indent=2, default=float))

    lines = ["| Scenario | Tickers | Trades | EV/trade | Sharpe (ann.) | DSR | Significant |",
             "|---|---|---|---|---|---|---|"]
    for k, r in sorted(prev_res.items()):
        lines.append(f"| {k} | {r['tickers']} | {fmt(r['n_trades'])} | {fmt(r['expected_value'], pct=True)} | "
                     f"{fmt(r['sr_annualized'])} | {fmt(r['dsr'])} | {'yes' if r['edge_is_significant'] else 'no'} |")
    (OUT / "results.md").write_text("# Massive re-validation\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

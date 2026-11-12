"""Re-test the live strategy (V4) on Massive data: does the edge survive survivorship bias?

    python -m research.massive.experiment [--only M0 M1 M2 M3] [--top-n 100]

Scenarios, declared 2026-11-12 before any of them was run. Same model and parameters as the
live pipeline (config.yaml, V4: momentum only, 20-session hold, live-execution barriers):
  M0 yfinance_live25   yfinance prices cut to the Massive window (baseline: same years)
  M1 massive_live25    Massive prices, the 25 live tickers (data check: should be near M0)
  M2 massive_pit       Massive, monthly point-in-time top-N by dollar volume INCLUDING
                       delisted stocks. THE decision scenario: only M2 with DSR >= 0.95 counts.
  M3 massive_news      M1 + news_sent_5d / news_count_5d meta-model features (research.massive.news)

Each scenario runs the full validation (make_signals: DSR, CPCV, crisis windows) and the
24-month walk-forward replay of the paper bot (research.replay.paper_replay.run_replay), and
is logged as a trial in reports/massive/trials_log.csv (seeded from the live log).
Outputs reports/massive/results.json and results.md. Raw Massive data stays in data/massive/.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from research.massive.universe import ALWAYS, DATA, common_stocks, load_grouped, pit_universe, split_reused_tickers
from research.replay.paper_replay import run_replay
from stocks_etl.features import make_features
from stocks_etl.stocks_etl import universe_symbols

OUT = Path("reports/massive")
CACHE = DATA / "features"
MASSIVE_START = "2016-11-14"  # first session the Developer plan's rolling 10 years allowed
NEWS_FEATURES = ["news_sent_5d", "news_count_5d"]


def features(name, df, qcfg, eligible=None):
    """GARCH/HMM per ticker is the slow part: cache per universe."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    print(f"[features] {name}: {df['ticker'].nunique()} tickers")
    feats, meta = make_features(df, qcfg)
    if eligible is not None:
        feats = feats.merge(eligible.assign(eligible=True), on=["date", "ticker"], how="left")
        feats["eligible"] = feats["eligible"].fillna(False).astype(bool)
        feats.loc[feats["ticker"].isin(ALWAYS), "eligible"] = True
    pd.to_pickle((feats, meta), path)
    return feats, meta


def run(tag, fm, prices, qcfg, cal, trials, **tags):
    feats, meta = fm
    cfg = dict(qcfg, **tags)
    print(f"\n=== {tag}: {feats['ticker'].nunique()} tickers, {tags} ===")
    s = run_replay(feats, meta, prices, cfg, cal, months=24, embargo=5, trials_path=str(trials), tag=tag)
    b = s["backtest"]
    return {"tickers": int(feats["ticker"].nunique()), **{k: b.get(k) for k in
            ("n_trades", "years", "win_rate", "expected_value", "sr_per_trade", "sr_annualized", "dsr", "n_trials")},
            "edge_is_significant": bool((b.get("dsr") or 0) > 0.95), "cpcv_path_sr": b["cpcv_path_sr"],
            "stress_windows": b.get("stress_windows"),
            "replay": {"gate0": s["gate0"]["status"], "closed": s["counts"]["closed"], "win_rate": s["win_rate"],
                       "expected_value": s["expected_value"], "sr_per_trade": s["sr_per_trade"],
                       "total_return": s["total_return"]}}


def fmt(v, pct=False, d=3):
    if v is None:
        return "-"
    return f"{v * 100:+.{d}f}%" if pct else (f"{v:,.{d}f}" if isinstance(v, float) else f"{v:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="stocks_etl/config.yaml")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--only", nargs="*", help="run a subset, e.g. --only M0 M1")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    qcfg = dict(cfg.get("quant") or {}, mc_paths=1000)
    cal = cfg.get("calendar")
    OUT.mkdir(parents=True, exist_ok=True)
    trials = OUT / "trials_log.csv"
    live_trials = Path(cfg.get("publish", {}).get("docs_dir", "docs/data")) / "history" / "trials_log.csv"
    if not trials.exists() and live_trials.exists():
        shutil.copy(live_trials, trials)
    # the replay variants (V1-V4) are attempts too: the DSR here must pay for them
    replay_trials = Path("research/replay/out/trials_log.csv")
    if replay_trials.exists():
        mine, rep = pd.read_csv(trials), pd.read_csv(replay_trials)
        pd.concat([mine, rep[~rep["config_hash"].isin(mine["config_hash"])]],
                  ignore_index=True).to_csv(trials, index=False)

    want = set(a.only or ["M0", "M1", "M2", "M3"])
    live25 = universe_symbols(cfg)
    start = pd.Timestamp(MASSIVE_START, tz="UTC")
    results = {}

    if "M0" in want:
        yf = pd.read_csv("research/replay/out/prices.csv.gz")
        yf["date"] = pd.to_datetime(yf["date"], utc=True)
        yf = yf[yf["date"] >= start].copy()
        results["M0 yfinance_live25"] = run("M0", features("yf_live25", yf, qcfg), yf, qcfg, cal, trials,
                                            data_source="yfinance", universe="live25", window_start=MASSIVE_START)
    if want & {"M1", "M3"}:
        surv = load_grouped(tickers=live25)
        fm = features("massive_live25", surv, qcfg)
        if "M1" in want:
            results["M1 massive_live25"] = run("M1", fm, surv, qcfg, cal, trials,
                                               data_source="massive", universe="live25")
        if "M3" in want:
            from research.massive.news import load_articles, news_features, score
            scored = score(load_articles([t for t in live25 if t not in ALWAYS]))
            feats = news_features(scored, fm[0])
            results["M3 massive_news"] = run("M3", (feats, fm[1]), surv, qcfg, cal, trials, data_source="massive",
                                             universe="live25", extra_features=NEWS_FEATURES)
    if "M2" in want:
        allp = split_reused_tickers(load_grouped())
        members = pit_universe(allp, common_stocks(), top_n=a.top_n)
        keep = set(members["ticker"]) | set(ALWAYS)
        pit = allp[allp["ticker"].isin(keep)].copy()
        print(f"[pit] {members['ticker'].nunique()} distinct stocks were ever in the monthly top {a.top_n}")
        fm = features(f"massive_pit_top{a.top_n}", pit, qcfg, eligible=members)
        results["M2 massive_pit"] = run("M2", fm, pit, qcfg, cal, trials, data_source="massive",
                                        universe=f"pit_top{a.top_n}")
        results["M2 massive_pit"]["skipped_short_history"] = int(pit["ticker"].nunique() - fm[0]["ticker"].nunique())

    prev = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    res = prev.get("results", {})
    res.update(results)
    run_meta = {"generated_utc": datetime.now(timezone.utc).isoformat(), "top_n": a.top_n,
                "massive_start": MASSIVE_START}
    (OUT / "results.json").write_text(json.dumps({"run": run_meta, "results": res}, indent=2, default=str))

    lines = ["| Scenario | Tickers | Trades | EV/trade | Sharpe (ann.) | DSR | Trials | Replay trades | "
             "Replay EV/trade | Replay return | Gate 0 |", "|" + "---|" * 11]
    for k, r in sorted(res.items()):
        p = r["replay"]
        lines.append(f"| {k} | {r['tickers']} | {fmt(r['n_trades'])} | {fmt(r['expected_value'], pct=True)} | "
                     f"{fmt(r['sr_annualized'], d=2)} | {fmt(r['dsr'])} | {r['n_trials']} | {p['closed']} | "
                     f"{fmt(p['expected_value'], pct=True)} | {fmt(p['total_return'], pct=True, d=2)} | {p['gate0']} |")
    (OUT / "results.md").write_text("# Massive re-test of V4\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

"""Pre-declared strategy variants, each backtested AND replayed, each logged as a trial.

    python -m research.replay.variants

The four variants below were written down before any of them was run (2026-11-11). They are
not tuned after the fact: changing them, or adding one, is a new trial the DSR must pay for.

  V1 honest_execution  backtest uses the live execution (next-open entry, intraday barriers,
                       gaps, double touch = stop). Strategy unchanged.
  V2 momentum_only     V1 without mean reversion (it lost -0.28%/trade in the replay).
  V3 closer_targets    V2 with k_tp = k_sl = 1.0 instead of 1.5 (72% of replay trades timed out).
  V4 longer_hold       V2 with a 20-session momentum time barrier instead of 10.

Trials go to research/replay/out/trials_log.csv, which starts as a copy of the live trials log,
so every DSR here is deflated for all live attempts plus every variant run so far.
Outputs: research/replay/out/variants.json and variants.md.
"""
import json
import shutil

import yaml

from research.replay.paper_replay import OUT, load_features, load_prices, run_replay

VARIANTS = {
    "V1_honest_execution": {"barrier_mode": "live"},
    "V2_momentum_only": {"barrier_mode": "live", "strategies": ["momentum"]},
    "V3_closer_targets": {"barrier_mode": "live", "strategies": ["momentum"], "k_tp": 1.0, "k_sl": 1.0},
    "V4_longer_hold": {"barrier_mode": "live", "strategies": ["momentum"], "momentum_horizon": 20},
}


def _pct(v, d=3):
    return "-" if v is None else f"{v * 100:+.{d}f}%"


def _num(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def main():
    cfg = yaml.safe_load(open("stocks_etl/config.yaml", encoding="utf-8"))
    qcfg = dict(cfg.get("quant") or {}, mc_paths=1000)
    cal = cfg.get("calendar")
    OUT.mkdir(parents=True, exist_ok=True)
    trials = OUT / "trials_log.csv"
    if not trials.exists():
        shutil.copy("docs/data/history/trials_log.csv", trials)

    prices = load_prices(cfg, False)
    feats, meta = load_features(prices, qcfg, False)

    results = {}
    for name, over in VARIANTS.items():
        print(f"\n=== {name}: {over} ===")
        s = run_replay(feats, meta, prices, dict(qcfg, **over), cal, trials_path=str(trials), tag=name)
        results[name] = {"overrides": over, "backtest": s["backtest"],
                         "replay": {"gate0": s["gate0"]["status"], "closed": s["counts"]["closed"],
                                    "win_rate": s["win_rate"], "expected_value": s["expected_value"],
                                    "sr_per_trade": s["sr_per_trade"], "total_return": s["total_return"],
                                    "exit_reasons": s["diagnostics"]["exit_reasons"]}}
        (OUT / "variants.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines = ["| Variant | Backtest trades | Backtest EV/trade | Backtest SR (ann.) | DSR | Trials | "
             "Replay trades | Replay EV/trade | Replay return | Gate 0 |", "|" + "---|" * 10]
    for name, r in results.items():
        b, p = r["backtest"], r["replay"]
        lines.append(f"| {name} | {b['n_trades']} | {_pct(b['expected_value'])} | {_num(b['sr_annualized'], 2)} | "
                     f"{_num(b['dsr'])} | {b['n_trials']} | {p['closed']} | {_pct(p['expected_value'])} | "
                     f"{_pct(p['total_return'], 2)} | {p['gate0']} |")
    (OUT / "variants.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

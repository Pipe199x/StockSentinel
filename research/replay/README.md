# Paper-trading replay (Gate 0 without waiting)

```
python -m research.replay.paper_replay --months 24      # --refresh to re-download / recompute
```

This replays the live paper-trading bot over the last N months. Each day it only sees what was knowable on that day:

| Piece | How look-ahead is avoided |
|---|---|
| Features | GARCH and HMM are refitted point-in-time every year (`tests/test_pit.py`), so they are computed once. |
| Meta-model | Retrained monthly on labels that were already resolved: `end < month start - embargo`. |
| Signals | Same sizing (½ Kelly, vol target, ES cap), take-profit, stop and exit date as `make_signals`. |
| Execution | `stocks_etl.paper.update_ledger` runs once per session, with prices truncated to that day. |

The output reports two executions of the same probabilities:
- **Paper execution.** Next-open entry and intraday barriers, as the live bot trades.
- **Backtest execution.** Entry at the signal close and barriers on closes, as `sequential_trades` does.

A large gap between the two comes from execution rules, not from the model.

The Gate 0 verdict uses `paper.summarize`. It compares against the CPCV range of a fresh validation run, which is not logged as a trial.

What the replay cannot test:
- Real broker fills and slippage.
- yfinance data revisions.
- Day-to-day operation of the cron job.

The live ledger on the dashboard keeps checking these.

Outputs go to `research/replay/out/` (gitignored):
- `replay_trades.csv`
- `replay_signals.csv`
- `replay_summary.json`

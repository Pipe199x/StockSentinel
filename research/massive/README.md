# Massive research month

Goal: re-validate the strategy with the two fixes free data can't give us, then cancel.

| Scenario | Universe | Barriers | Question |
|---|---|---|---|
| S0 | 25 live tickers | daily closes | Does Massive data reproduce the published numbers? |
| S1 | 25 live tickers | intraday high/low + minute tie-break | Do stops hit inside the day change the result? |
| S2 | monthly top-100 by dollar volume, **incl. delisted** | daily closes | How much was survivorship bias flattering us? |
| S3 | same as S2 | intraday | The honest answer. |

Every scenario is logged as a trial (`reports/massive/trials_log.csv`, seeded from the live log),
so the Deflated Sharpe pays for these attempts. Decision rule: only S3 with DSR >= 0.95 is evidence of an edge.

## Before paying: test with a free key
1. Create a free Massive account, put the key in `.env` as `MASSIVE_API_KEY=...` (gitignored).
2. `python -m research.massive.download --start <one month ago> --max-per-min 5` (about 5 minutes).

## During the paid month
```bash
python -m research.massive.download                  # ~10 years, ~2,500 requests, resumable
python -m research.massive.experiment --only S0      # sanity check first
python -m research.massive.experiment                # all four; features are the slow part
```
Results: `reports/massive/results.md` and `results.json`. Raw data: `data/massive/` (gitignored, keep private).

## Known limits
- A symbol that is reused by another company after a >10-session gap is split into `TICKER~1`;
  renames (e.g. FB -> META) look like a delisting plus a new listing.
- Delisted trades exit at the last exchange close; real post-delisting (OTC) prices are often lower.
- Prices are split-adjusted, not dividend-adjusted (as the live pipeline was then; since 2026-11-14 it adds cash dividends to trade returns).

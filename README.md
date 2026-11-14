<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="web/public/brand/odg-badge-white.svg">
    <img src="web/public/brand/odg-badge-navy.svg" alt="Ocean Data Group" width="260">
  </picture>
</p>

<h1 align="center">StockSentinel</h1>

<p align="center"><b>It already decided. You just haven't seen it yet.</b></p>

---

Every night, while you sleep, something reads the market (25 assets, the headlines,
the mood underneath it all) and makes up its mind.

It doesn't hesitate. It doesn't get attached. It doesn't explain itself to strangers.

By morning the choice is made. A few people know what it is.

## For the nerds

- **Data:** Massive (prices, dividends, splits, news). Complete sessions only, point-in-time features, no look-ahead.
- **Gate:** a hidden Markov model on the broad market. In stress, it does nothing at all.
- **Edge hunt:** a momentum rule proposes; an XGBoost meta-model with FinBERT news sentiment disposes. Labels from the triple barrier.
- **Sizing:** ½ Kelly × volatility target, capped by Expected Shortfall.
- **Honesty:** purged K-fold + CPCV, and a Deflated Sharpe that charges for every version ever tried. A paper ledger fills every call at the next open and keeps the score.
- **Plumbing:** GitHub Actions → static JSON → Next.js 16 on Cloudflare Pages. The interesting part sits behind a Pages Function that only opens for a valid license key.

## And then

Today's answer is already published.<br>
Yesterday's is already playing out.<br>
Tomorrow's is being computed tonight.

You can keep reading READMEs.

**→ [trading.oceandatagroup.ai](https://trading.oceandatagroup.ai)**

---

<sub>Model portfolio; results are hypothetical. Not investment advice.</sub>

<p align="center"><sub>© Ocean Data Group</sub></p>

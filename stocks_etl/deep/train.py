"""Entrenamiento semanal de los meta-modelos deep, con compuerta de promoción.

    python -m stocks_etl.deep.train --config stocks_etl/config.yaml

Compara LSTM y fusión (precio + noticias) contra el XGBoost diario sobre los MISMOS
eventos y con el mismo Purged K-Fold. Un modelo deep solo se promueve si su Deflated
Sharpe supera 0.95 y además mejora al XGBoost; si no, queda "en evaluación". Cada
modelo probado se registra en trials_log.csv: también es un intento.
"""
import argparse
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from stocks_etl.deep.data import STATIC, build_sequences, load_news, news_coverage, news_tensor, standardize
from stocks_etl.deep.models import FusionMeta, LSTMMeta, fit_predict
from stocks_etl.features import make_features
from stocks_etl.meta_model import FEATURES, _model, _trade_stats, build_events, sequential_trades
from stocks_etl.publish_json import write_json
from stocks_etl.validation import PurgedKFold, deflated_sharpe, log_trial, read_trials


def _oof(ev: pd.DataFrame, cfg: dict, predict) -> np.ndarray:
    """Probabilidades fuera de muestra con Purged K-Fold + embargo."""
    oof = np.full(len(ev), np.nan)
    starts, ends = ev["start"].to_numpy(), ev["end"].to_numpy()
    for tr, te in PurgedKFold(int(cfg.get("n_splits", 5)), float(cfg.get("embargo_pct", 0.01))).split(starts, ends):
        if len(tr) >= 50 and len(te):
            oof[te] = predict(tr, te)
    return oof


def _score(ev: pd.DataFrame, prob: np.ndarray, cfg: dict, trial_cfg: Optional[dict], trials_path: str,
           as_of: str) -> Dict:
    """trial_cfg=None: línea base ya registrada (solo lee el registro, no cuenta como intento)."""
    trades = sequential_trades(ev, prob, cfg)
    stats = _trade_stats(trades)
    if stats.get("n_trades", 0) > 1:
        if trial_cfg is None:
            n_trials, var_sr = read_trials(trials_path)
        else:
            n_trials, var_sr = log_trial(trials_path, trial_cfg, stats["sr_per_trade"], stats["n_trades"], as_of)
        stats["n_trials"] = n_trials
        stats["dsr"] = deflated_sharpe(stats["sr_per_trade"], n_trials, var_sr, stats["skew"], stats["kurtosis"],
                                       stats["n_trades"])
    return stats


def _promote(deep: Dict, base: Dict) -> bool:
    return bool(np.isfinite(deep.get("dsr", np.nan)) and deep["dsr"] > 0.95
                and deep.get("sr_per_trade", -np.inf) > base.get("sr_per_trade", np.inf))


def main(cfg: dict) -> Dict:
    qcfg = cfg.get("quant", {}) or {}
    dcfg = cfg.get("deep", {}) or {}
    docs = (cfg.get("publish") or {}).get("docs_dir", "docs/data")
    trials_path = os.path.join(docs, "history", "trials_log.csv")
    seed = int(qcfg.get("seed", 42))
    lookback = int(dcfg.get("lookback", 20))
    epochs = int(dcfg.get("epochs", 8))
    hidden = int(dcfg.get("hidden", 32))

    df = pd.read_csv(os.path.join(docs, "history", "prices_history.csv"), parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    feats, meta = make_features(df, qcfg)
    ev, latest = build_events(feats, qcfg)
    as_of = str(feats["date"].max().date())

    X, ok = build_sequences(feats, ev[["ticker", "start"]], lookback)
    ev, X = ev[ok].reset_index(drop=True), X[ok]
    S = ev[STATIC].to_numpy(dtype=np.float32)
    y = ev["label"].to_numpy()
    print(f"[DEEP] events={len(ev)} lookback={lookback}")

    base_trial = {k: qcfg.get(k) for k in sorted(qcfg) if k != "mc_paths"}
    deep_trial = {"lookback": lookback, "epochs": epochs, "hidden": hidden}

    def xgb_predict(tr, te, rows=None):
        rows = ev if rows is None else rows
        m = _model(seed).fit(rows[FEATURES].iloc[tr], rows["label"].iloc[tr])
        return m.predict_proba(rows[FEATURES].iloc[te])[:, 1]

    def lstm_predict(tr, te):
        return fit_predict(LSTMMeta, standardize(X, tr), S, y, tr, te, epochs=epochs, seed=seed, hidden=hidden)

    xgb_oof = _oof(ev, qcfg, xgb_predict)
    report = {"as_of": as_of, "generated_utc": datetime.now(timezone.utc).isoformat(), "n_events": int(len(ev))}
    # el XGB ya está registrado por el ETL diario con esta misma configuración: no es un intento nuevo
    report["xgb"] = _score(ev, xgb_oof, qcfg, None, trials_path, as_of)
    report["lstm"] = _score(ev, _oof(ev, qcfg, lstm_predict), qcfg, {**base_trial, "model": "lstm", **deep_trial},
                            trials_path, as_of)
    report["lstm"]["promoted"] = _promote(report["lstm"], report["xgb"])
    print(f"[DEEP] xgb SR/trade={report['xgb'].get('sr_per_trade')} lstm SR/trade={report['lstm'].get('sr_per_trade')} "
          f"DSR={report['lstm'].get('dsr')}")

    # --- fusión precio + noticias: solo donde hay noticias con embedding ----------------
    news = load_news(os.path.join(docs, "history", "news_history.ndjson"),
                     os.path.join(docs, "history", "news_embeddings.npz"))
    cov = news_coverage(ev[["ticker", "start"]], news)
    min_events = int(dcfg.get("min_news_events", 500))
    fusion: Dict = {"events_with_news_coverage": int(cov.sum()), "min_events": min_events,
                    "news_tickers": sorted(set(news["ticker"])) if news is not None else [],
                    "news_since": str(news["published"].min().date()) if news is not None else None}
    if cov.sum() < min_events:
        fusion["status"] = "insufficient_data"
        fusion["promoted"] = False
        print(f"[DEEP] fusion skipped: {int(cov.sum())} events with news < {min_events}")
    else:
        sub = np.flatnonzero(cov)
        evs, Xs, Ss, ys = ev.iloc[sub].reset_index(drop=True), X[sub], S[sub], y[sub]
        E, M = news_tensor(evs[["ticker", "start"]], news, int(dcfg.get("news_days", 5)), int(dcfg.get("news_items", 8)))

        def fusion_predict(tr, te):
            return fit_predict(FusionMeta, standardize(Xs, tr), Ss, ys, tr, te, E=E, M=M, epochs=epochs, seed=seed,
                               hidden=hidden)

        fusion["xgb_same_events"] = _score(evs, _oof(evs, qcfg, lambda tr, te: xgb_predict(tr, te, evs)), qcfg,
                                           None, trials_path, as_of)
        fusion.update(_score(evs, _oof(evs, qcfg, fusion_predict), qcfg,
                             {**base_trial, "model": "fusion", **deep_trial,
                              "news_days": dcfg.get("news_days", 5), "news_items": dcfg.get("news_items", 8)},
                             trials_path, as_of))
        fusion["status"] = "evaluated"
        fusion["promoted"] = _promote(fusion, fusion["xgb_same_events"])
    report["fusion"] = fusion

    # --- señales del modelo promovido (si lo hay) --------------------------------------
    signals = []
    if report["lstm"]["promoted"]:
        live = latest[latest["side"] != 0].copy()
        live["start"] = live["date"].dt.tz_localize(None)
        Xl, okl = build_sequences(feats, live[["ticker", "start"]], lookback)
        live, Xl = live[okl].reset_index(drop=True), Xl[okl]
        if len(live):
            Xa = np.concatenate([X, Xl])
            Sa = np.concatenate([S, live[STATIC].to_numpy(dtype=np.float32)])
            ya = np.concatenate([y, np.zeros(len(live), dtype=y.dtype)])
            tr, te = np.arange(len(X)), np.arange(len(X), len(Xa))
            p = fit_predict(LSTMMeta, standardize(Xa, tr), Sa, ya, tr, te, epochs=epochs, seed=seed, hidden=hidden)
            signals = [{"ticker": t, "as_of": as_of, "side": int(s), "p_success_lstm": float(v)}
                       for t, s, v in zip(live["ticker"], live["side"], p)]
    report["signals"] = signals
    report["promoted_model"] = "lstm" if report["lstm"]["promoted"] else "fusion" if fusion.get("promoted") else None
    write_json(report, os.path.join(docs, "deep_report.json"))
    print(f"[DEEP] promoted={report['promoted_model']} -> {os.path.join(docs, 'deep_report.json')}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stocks_etl/config.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        main(yaml.safe_load(f))

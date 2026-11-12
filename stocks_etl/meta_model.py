"""Meta-labeling + validación honesta + señales del día.

Capa 1 (primary.py): dice el lado (comprar / vender / nada).
Capa 2 (aquí): un XGBClassifier estima P(la señal primaria acierta tras costos) y esa
probabilidad define el tamaño con Kelly fraccionado. Toda la evaluación es fuera de
muestra con Purged K-Fold + embargo y CPCV; el Deflated Sharpe castiga los intentos.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from xgboost import XGBClassifier

from stocks_etl.features import MODEL_FEATURES
from stocks_etl.labeling import triple_barrier, triple_barrier_live, triple_barrier_ohlc
from stocks_etl.primary import STRATEGY_NAMES, primary_side
from stocks_etl.regime import REGIME_NAMES
from stocks_etl.risk import (
    cornish_fisher_var, es_cap, expected_max_losing_streak, kelly_fraction, max_losing_streak,
    mc_t_paths, streak_drawdown, var_es_t, vol_target_scale,
)
from stocks_etl.validation import (
    PurgedKFold, cpcv_paths, cpcv_splits, deflated_sharpe, log_trial, min_sample_size, sharpe,
)

FEATURES = MODEL_FEATURES + ["side", "strategy"]


def feature_cols(cfg: dict) -> List[str]:
    """Meta-model inputs: the fixed list plus research-only extras (cfg['extra_features'])."""
    return FEATURES + [c for c in (cfg.get("extra_features") or []) if c not in FEATURES]


def _model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, random_state=seed, n_jobs=1, eval_metric="logloss",
    )


def build_events(feats: pd.DataFrame, cfg: dict, resolver=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Une señal primaria + triple barrera. Devuelve (eventos etiquetados, última fila por ticker).

    Opciones de investigación (por defecto apagadas, el pipeline diario no cambia):
      - cfg['barrier_mode'] == 'intraday': barreras sobre máximos/mínimos (triple_barrier_ohlc);
        ``resolver(ticker, date, tp_price, sl_price, side)`` desempata días con doble toque.
      - columna ``eligible`` en feats: solo se generan eventos en días en que el ticker
        pertenecía al universo point-in-time (sin sesgo de supervivencia).
    """
    k_tp, k_sl = float(cfg.get("k_tp", 1.5)), float(cfg.get("k_sl", 1.5))
    cost = 2 * float(cfg.get("cost_bps", 5)) / 1e4
    intraday = cfg.get("barrier_mode", "close") == "intraday"
    global_last = feats["date"].max()
    events, latest = [], []
    for tkr, f in feats.groupby("ticker", sort=False):
        f = f.sort_values("date").reset_index(drop=True)
        f = pd.concat([f, primary_side(f, cfg)], axis=1)
        f["sigma_event"] = f["sigma_garch"] * np.sqrt(f["t_max"])
        latest.append(f.iloc[[-1]])
        mask = f["side"].to_numpy() != 0
        if "eligible" in f.columns:
            mask &= f["eligible"].fillna(False).to_numpy(dtype=bool)
        pos = np.flatnonzero(mask)
        if len(pos) == 0:
            continue
        args = (pos, f["side"].to_numpy()[pos], f["sigma_event"].to_numpy()[pos], k_tp, k_sl,
                f["t_max"].to_numpy()[pos])
        if cfg.get("barrier_mode") == "live":
            tb = triple_barrier_live(f["open"].to_numpy(), f["high"].to_numpy(), f["low"].to_numpy(),
                                     f["close"].to_numpy(), *args,
                                     series_ended=bool(f["date"].iloc[-1] < global_last))
        elif intraday:
            dates = f["date"].to_numpy()
            resolve = None
            if resolver is not None:
                resolve = (lambda p, tp_px, sl_px, s, _t=tkr, _d=dates:
                           resolver(_t, pd.Timestamp(_d[p]), tp_px, sl_px, s))
            tb = triple_barrier_ohlc(f["open"].to_numpy(), f["high"].to_numpy(), f["low"].to_numpy(),
                                     f["close"].to_numpy(), *args, resolve=resolve,
                                     series_ended=bool(f["date"].iloc[-1] < global_last))
        else:
            tb = triple_barrier(f["close"].to_numpy(), *args)
        ev = f.iloc[pos].reset_index(drop=True)
        ev["start"] = ev["date"]
        ev["end"] = f["date"].to_numpy()[tb["end_pos"].to_numpy()]
        ev["ret"] = tb["ret"].to_numpy()
        ev["barrier"] = tb["barrier"].to_numpy()
        if "ambiguous" in tb:
            ev["ambiguous"] = tb["ambiguous"].to_numpy()
        events.append(ev)
    ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    if not ev.empty:
        ev = ev[ev["barrier"] != "open"].copy()
        ev["start"] = ev["start"].dt.tz_localize(None)
        ev["end"] = pd.to_datetime(ev["end"]).dt.tz_localize(None)
        ev["ret_net"] = ev["ret"] - cost
        ev["label"] = (ev["ret_net"] > 0).astype(int)
        ev = ev.sort_values(["start", "ticker"]).reset_index(drop=True)
    return ev, pd.concat(latest, ignore_index=True)


def position_size(p: float, sigma: float, nu: float, t_max: float, cfg: dict) -> Tuple[float, str]:
    """½ Kelly × objetivo de volatilidad, recortado para que el Expected Shortfall (1%) de la
    operación no cueste más de `max_es_loss` del capital. Devuelve (tamaño, límite activo)."""
    payoff = float(cfg.get("k_tp", 1.5)) / float(cfg.get("k_sl", 1.5))
    size = kelly_fraction(p, payoff, float(cfg.get("kelly_fraction", 0.5))) *         vol_target_scale(sigma, float(cfg.get("target_vol", 0.20)))
    max_loss = cfg.get("max_es_loss")
    if max_loss and size > 0:
        cap = es_cap(sigma, nu, t_max, float(max_loss))
        if cap < size:
            return cap, "es"
    return size, "kelly"


STRESS_WINDOWS = {
    "GFC 2008": ("2007-10-09", "2009-03-09"),
    "COVID crash": ("2020-02-19", "2020-04-30"),
    "2022 bear market": ("2022-01-03", "2022-10-12"),
    "April 2025 sell-off": ("2025-04-02", "2025-04-30"),
}


def _stress_windows(trades: pd.DataFrame, feats: pd.DataFrame, bench: str) -> Dict:
    """Operaciones fuera de muestra que empezaron dentro de cada crisis, y el retorno del
    benchmark en ese periodo como contexto. Una ventana sin datos se marca covered=False."""
    spy = feats.loc[feats["ticker"] == bench, ["date", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None)
    spy = spy.set_index("date")["close"]
    out = {}
    for name, (a, b) in STRESS_WINDOWS.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        px = spy[(spy.index >= a) & (spy.index <= b)]
        row = {"from": str(a.date()), "to": str(b.date()), "covered": bool(len(px) > 1),
               "benchmark_return": float(px.iloc[-1] / px.iloc[0] - 1) if len(px) > 1 else None,
               "n_trades": 0, "win_rate": None, "expected_value": None, "pnl_sum": 0.0}
        if not trades.empty:
            t = trades[(trades["start"] >= a) & (trades["start"] <= b)]
            if len(t):
                wins = t["ret_net"] > 0
                row.update({"n_trades": int(len(t)), "win_rate": float(wins.mean()),
                            "expected_value": float(t["ret_net"].mean()), "pnl_sum": float(t["pnl"].sum())})
        out[name] = row
    return out


def sequential_trades(ev: pd.DataFrame, prob: np.ndarray, cfg: dict, use_meta: bool = True) -> pd.DataFrame:
    """Backtest sin solapamiento: por ticker, una posición a la vez hasta tocar barrera.

    Con meta-modelo: se opera si p > 0.5 y el tamaño es Kelly fraccionado × objetivo de
    volatilidad. Sin él (línea base primaria): se opera todo con tamaño 1.
    """
    rows = []
    d = ev.assign(prob=prob)
    if "garch_nu" not in d:
        d["garch_nu"] = np.nan
    for tkr, g in d.groupby("ticker", sort=False):
        busy_until = None
        for r in g.sort_values("start").itertuples():
            if busy_until is not None and r.start <= busy_until:
                continue
            if use_meta:
                if not np.isfinite(r.prob) or r.prob <= 0.5:
                    continue
                size, _ = position_size(r.prob, r.sigma_garch, r.garch_nu, r.t_max, cfg)
                if size <= 0:
                    continue
            else:
                size = 1.0
            rows.append({"ticker": tkr, "start": r.start, "end": r.end, "side": r.side, "strategy": r.strategy,
                         "regime": r.mkt_regime, "size": size, "ret_net": r.ret_net, "pnl": size * r.ret_net})
            busy_until = r.end
    return pd.DataFrame(rows)


def _trade_stats(trades: pd.DataFrame) -> Dict:
    if trades.empty:
        return {"n_trades": 0}
    pnl = trades["pnl"].to_numpy()
    years = max((trades["end"].max() - trades["start"].min()).days / 365.25, 1e-9)
    sr = sharpe(pnl)
    wins = (trades["ret_net"] > 0).to_numpy()
    win_rate = float(wins.mean())
    avg_win = float(trades.loc[wins, "ret_net"].mean()) if wins.any() else 0.0
    avg_loss = float(-trades.loc[~wins, "ret_net"].mean()) if (~wins).any() else 0.0
    return {
        "n_trades": int(len(trades)),
        "years": round(years, 2),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        # esperanza matemática por trade: P(ganar)·ganancia media - P(perder)·pérdida media
        "expected_value": win_rate * avg_win - (1 - win_rate) * avg_loss,
        "sr_per_trade": sr,
        "sr_annualized": sr * math.sqrt(len(trades) / years),
        "skew": float(skew(pnl)) if len(pnl) > 2 else 0.0,
        "kurtosis": float(kurtosis(pnl, fisher=False)) if len(pnl) > 3 else 3.0,
        "max_losing_streak": max_losing_streak(wins),
    }


def evaluate(ev: pd.DataFrame, cfg: dict) -> Tuple[Dict, np.ndarray]:
    """Purged K-Fold (OOF) + CPCV. Devuelve (informe, probabilidades OOF)."""
    seed = int(cfg.get("seed", 42))
    X, y = ev[feature_cols(cfg)], ev["label"].to_numpy()
    starts, ends = ev["start"].to_numpy(), ev["end"].to_numpy()
    emb = float(cfg.get("embargo_pct", 0.01))

    oof = np.full(len(ev), np.nan)
    folds = []
    for tr, te in PurgedKFold(int(cfg.get("n_splits", 5)), emb).split(starts, ends):
        m = _model(seed).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        fold_trades = sequential_trades(ev.iloc[te], oof[te], cfg)
        folds.append({
            "test_from": str(pd.Timestamp(starts[te[0]]).date()),
            "test_to": str(pd.Timestamp(starts[te[-1]]).date()),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "accuracy": float(((oof[te] > 0.5) == y[te]).mean()),
            "sr_per_trade": sharpe(fold_trades["pnl"]) if not fold_trades.empty else 0.0,
        })

    # CPCV: muchos caminos históricos en lugar de uno solo -> distribución de Sharpe
    n_groups, k_test = int(cfg.get("cpcv_groups", 6)), int(cfg.get("cpcv_k", 2))
    split_results = []
    for tr, te, test_groups, group_of in cpcv_splits(starts, ends, n_groups, k_test, emb):
        m = _model(seed).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        per_group = {g: (te[group_of[te] == g], p[group_of[te] == g]) for g in test_groups}
        split_results.append((test_groups, per_group))
    path_srs = []
    for path in cpcv_paths(split_results, n_groups, k_test):
        idx = np.concatenate([path[g][0] for g in sorted(path)])
        prob = np.concatenate([path[g][1] for g in sorted(path)])
        t = sequential_trades(ev.iloc[idx], prob, cfg)
        path_srs.append(sharpe(t["pnl"]) if not t.empty else 0.0)

    meta_trades = sequential_trades(ev, oof, cfg, use_meta=True)
    base_trades = sequential_trades(ev, np.ones(len(ev)), cfg, use_meta=False)
    return {"folds": folds, "cpcv_path_sr": path_srs,
            "meta": _trade_stats(meta_trades), "primary": _trade_stats(base_trades),
            "_meta_trades": meta_trades}, oof


def _nth_trading_day(as_of: pd.Timestamp, n: int, cal_name: Optional[str]) -> str:
    """n-ésima sesión bursátil estrictamente posterior a as_of (festivos excluidos)."""
    start = (as_of + pd.Timedelta(days=1)).date()
    end = (as_of + pd.Timedelta(days=n * 2 + 14)).date()
    if cal_name:
        try:
            import pandas_market_calendars as mcal
            sched = mcal.get_calendar(cal_name).schedule(start_date=start, end_date=end)
            if len(sched) >= n:
                return sched.index[n - 1].strftime("%Y-%m-%d")
        except Exception:
            pass
    return (as_of + pd.offsets.BDay(n)).strftime("%Y-%m-%d")


def _risk_row(f_hist: pd.DataFrame, garch: Dict, cfg: dict) -> Dict:
    sigma, nu, mu = garch["sigma_next"], garch["nu"], garch["mu"]
    r = f_hist["log_ret_1"].dropna().iloc[-252:]
    v95 = var_es_t(sigma, nu, 0.05, mu)
    v99 = var_es_t(sigma, nu, 0.01, mu)
    horizon = int(cfg.get("mc_horizon", 20))
    paths = mc_t_paths(mu, sigma, nu, horizon, int(cfg.get("mc_paths", 10000)), int(cfg.get("seed", 42)))
    return {
        "sigma_garch": sigma,
        "garch_persistence": garch["persistence"],
        "garch_nu": nu,
        "var_95": v95["var"],
        "es_99": v99["es"],
        "var_95_cornish_fisher": cornish_fisher_var(sigma, float(skew(r)), float(kurtosis(r)), 0.05, mu),
        "mc_horizon_days": horizon,
        "mc_p05": float(np.percentile(paths, 5)),
        "mc_p50": float(np.percentile(paths, 50)),
        "mc_p95": float(np.percentile(paths, 95)),
        "mc_prob_loss_10pct": float((paths <= -0.10).mean()),
    }


def make_signals(feats: pd.DataFrame, meta: Dict[str, Dict], cfg: dict, cal_name: Optional[str] = None,
                 trials_path: Optional[str] = None, resolver=None) -> Tuple[pd.DataFrame, Dict]:
    """Punto de entrada: devuelve (señales del día por ticker, informe de validación)."""
    seed = int(cfg.get("seed", 42))
    ev, latest = build_events(feats, cfg, resolver=resolver)
    if len(ev) < 200:
        raise SystemExit(f"Solo {len(ev)} eventos etiquetados: muestra insuficiente para validar")
    report, oof = evaluate(ev, cfg)
    meta_trades = report.pop("_meta_trades")

    cols = feature_cols(cfg)
    final = _model(seed).fit(ev[cols], ev["label"])
    importance = dict(sorted(zip(cols, final.feature_importances_.astype(float)), key=lambda kv: -kv[1]))

    payoff = float(cfg.get("k_tp", 1.5)) / float(cfg.get("k_sl", 1.5))
    k_tp, k_sl = float(cfg.get("k_tp", 1.5)), float(cfg.get("k_sl", 1.5))
    rows = []
    for r in latest.itertuples():
        as_of = pd.Timestamp(r.date)
        last_close = float(r.close)
        side = int(r.side)
        p = None
        size, limit = 0.0, None
        if side != 0 and pd.notna(r.sigma_event):
            x = pd.DataFrame([{c: getattr(r, c) for c in cols}])
            p = float(final.predict_proba(x)[0, 1])
            nu = getattr(r, "garch_nu", np.nan)
            size, limit = position_size(p, r.sigma_garch, nu if pd.notna(nu) else meta[r.ticker]["garch"]["nu"],
                                        r.t_max, cfg)
        regime = int(r.regime) if pd.notna(r.regime) else None
        row = {
            "ticker": r.ticker,
            "as_of": as_of.strftime("%Y-%m-%d"),
            "last_close": last_close,
            "regime": REGIME_NAMES.get(regime),
            "p_stress": float(r.p_stress) if pd.notna(r.p_stress) else None,
            "market_regime": REGIME_NAMES.get(int(r.mkt_regime)) if pd.notna(r.mkt_regime) else None,
            "market_p_stress": float(r.mkt_p_stress) if pd.notna(r.mkt_p_stress) else None,
            "hurst": float(r.hurst) if pd.notna(r.hurst) else None,
            "half_life": float(r.half_life) if pd.notna(r.half_life) else None,
            "zscore": float(r.zscore) if pd.notna(r.zscore) else None,
            "strategy": STRATEGY_NAMES[int(r.strategy)],
            "side": side,
            "p_success": p,
            "kelly_size": round(size, 4),
            "size_limit": limit if size > 0 else None,
            "tp": last_close * (1 + side * k_tp * r.sigma_event) if side else None,
            "sl": last_close * (1 - side * k_sl * r.sigma_event) if side else None,
            "t_max_days": int(r.t_max) if side else None,
            "t_max_date": _nth_trading_day(as_of, int(r.t_max), cal_name) if side else None,
        }
        row.update(_risk_row(feats[feats["ticker"] == r.ticker], meta[r.ticker]["garch"], cfg))
        rows.append(row)
    signals = pd.DataFrame(rows)

    m = report["meta"]
    model_cfg = {k: cfg.get(k) for k in sorted(cfg) if k not in ("mc_paths",)}
    n_trials, var_sr_trials = (log_trial(trials_path, model_cfg, m.get("sr_per_trade", 0.0), m.get("n_trades", 0),
                                         str(feats["date"].max().date())) if trials_path else (1, 0.0))
    var_sr = var_sr_trials if n_trials > 1 and var_sr_trials > 0 else float(np.var(report["cpcv_path_sr"], ddof=1))
    if m.get("n_trades", 0) > 1:
        m["dsr"] = deflated_sharpe(m["sr_per_trade"], n_trials, var_sr, m["skew"], m["kurtosis"], m["n_trades"])
        m["expected_streak"] = expected_max_losing_streak(m["win_rate"], m["n_trades"])
        m["streak_drawdown_1pct"] = streak_drawdown(m["expected_streak"], 0.01)
        m["streak_drawdown_5pct"] = streak_drawdown(m["expected_streak"], 0.05)
        m["min_trades_for_edge"] = min_sample_size(max(m["win_rate"], 0.501))

    by_regime = meta_trades.groupby("regime").size() if not meta_trades.empty else pd.Series(dtype=int)
    report.update({
        "n_events": int(len(ev)),
        "n_trials": n_trials,
        "var_sr_used": var_sr,
        "dsr_threshold": 0.95,
        "edge_is_significant": bool(m.get("dsr", 0) > 0.95),
        "trades_by_regime": {REGIME_NAMES.get(int(k), str(k)): int(v) for k, v in by_regime.items()},
        # el estrés no se opera por diseño; la cobertura que importa es la de los regímenes operados
        "regimes_undercovered": [REGIME_NAMES[k] for k in (0, 1) if int(by_regime.get(k, 0)) < 30],
        "feature_importance": importance,
        "correlation": _correlation_regimes(feats, cfg.get("benchmark", "SPY")),
        "stress_windows": _stress_windows(meta_trades, feats, cfg.get("benchmark", "SPY")),
        "config": model_cfg,
        # lo que esta validación NO cubre; se publica para no vender un backtest perfecto
        "limitations": [
            ("Survivorship bias: the live universe is 25 companies alive today. A point-in-time test including delisted "
             "stocks (research/massive, M2) did not reach DSR 0.95."),
            "History starts on 2016-11-14 (Massive plan window): the 2008 crisis is not tested.",
            ("GARCH and HMM parameters are refitted once a year using only earlier data (point-in-time); "
             "the first %d sessions are training only and are never traded." % int(cfg.get("pit_min_train", 500)))
            if cfg.get("pit_refit") else
            "GARCH and HMM parameters are fitted on the full history; the volatility and regime series themselves are filtered (past-only).",
            "The per-asset HMM can switch between two near-equally likely fits as data arrives; only the market (SPY) regime gates trading, and it is stable.",
            "Daily bars only: no volume bars, order book or intraday sessions.",
            ("Trades enter at the next session's open and barriers use intraday highs/lows, exactly like the paper ledger; a day that touches both counts as the stop." if cfg.get("barrier_mode") == "live" else "Barriers are checked on daily closes, not intraday highs/lows."),
            "With few trials the DSR is close to a Probabilistic Sharpe Ratio against zero.",
        ],
    })
    return signals, report


def _correlation_regimes(feats: pd.DataFrame, bench: str) -> Dict:
    """Correlación media entre activos en días normales vs días de estrés del mercado.

    En crisis las correlaciones se disparan: la diversificación que importa es la de ese día.
    """
    wide = feats.pivot(index="date", columns="ticker", values="log_ret_1").dropna(how="all")

    def avg_corr(w: pd.DataFrame) -> Optional[float]:
        if len(w) < 20:
            return None
        c = w.corr().to_numpy()
        return float(c[~np.eye(len(c), dtype=bool)].mean())

    out = {"all_days": avg_corr(wide)}
    if bench in wide.columns:
        reg = feats[feats["ticker"] == bench].set_index("date")["regime"].reindex(wide.index)
        out["market_stress_days"] = avg_corr(wide[reg == 2])
        out["market_calm_days"] = avg_corr(wide[reg == 0])
        out["n_stress_days"] = int((reg == 2).sum())
    return out

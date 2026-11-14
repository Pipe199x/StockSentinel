# stocks_etl/stocks_etl.py — daily ETL; prices from Massive only
import argparse
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
import pandas as pd
import pandas_market_calendars as mcal

from stocks_etl import advisor_report
from stocks_etl.meta_model import make_signals
from stocks_etl.paper import run_paper
from stocks_etl.pipeline import build_feature_frame
from stocks_etl.publish_json import (
    write_prices_latest_json,
    write_prices_history_csv,
    write_json,
    write_public_and_premium,
    redact_signals,
    redact_paper,
    append_signals_history,
)
from etl_common.corporate_actions import refresh_dividends
from etl_common.manifest import update_manifest
from etl_common.news_features import CLOSE_UTC_HOUR
from etl_common.sessions import drop_incomplete_sessions
from etl_common.validation import validate_prices

warnings.simplefilter("ignore", FutureWarning)


def _utc_now(now_utc=None) -> datetime:
    """One clock for the whole run (overridable with --now for reproducible runs)."""
    if now_utc is None:
        return datetime.now(timezone.utc)
    ts = pd.Timestamp(now_utc)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def write_single_prices_csv(df, base_dir, compression=None, now=None):
    """Escribe un SOLO CSV combinado con todos los tickers."""
    if df.empty:
        return []
    os.makedirs(base_dir, exist_ok=True)
    date_tag = _utc_now(now).strftime("%Y%m%d")
    filename = f"prices_{date_tag}.csv" + (".gz" if compression == "gzip" else "")
    path = os.path.join(base_dir, filename)
    df.to_csv(path, index=False, compression=compression)
    return [path]


def write_signals_csv(signals, base_dir, now=None):
    os.makedirs(base_dir, exist_ok=True)
    date_tag = _utc_now(now).strftime("%Y%m%d")
    path = os.path.join(base_dir, f"signals_{date_tag}.csv")
    signals.to_csv(path, index=False)
    return [path]


def universe_symbols(cfg):
    """Lista plana de tickers desde universe: {sector: [tickers]} (o symbols, formato antiguo)."""
    if cfg.get("universe"):
        return [t for tickers in cfg["universe"].values() for t in tickers]
    return list(cfg["symbols"])


def _trading_days(df, cal_name):
    """Keep only NYSE sessions (drops any stray non-session bar)."""
    if df.empty or not cal_name:
        return df
    sched = mcal.get_calendar(cal_name).schedule(start_date=df["date"].min().date(),
                                                  end_date=df["date"].max().date())
    valid = pd.DatetimeIndex(pd.to_datetime(sched.index)).tz_localize("UTC").normalize()
    return df[df["date"].isin(valid)]


def massive_frame(rows, ticker):
    """Massive daily aggregates -> the pipeline schema (date = session day at 00:00 UTC)."""
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    day = pd.to_datetime(d["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return pd.DataFrame({"date": day.dt.normalize().dt.tz_localize("UTC"), "ticker": ticker,
                         "open": d["o"].astype(float), "high": d["h"].astype(float), "low": d["l"].astype(float),
                         "close": d["c"].astype(float), "adj_close": pd.NA, "volume": d["v"].astype(float),
                         "dividends": 0.0, "stock_splits": 0.0})


def _history_start(cfg, end: str) -> pd.Timestamp:
    years = int(cfg.get("massive_history_years", 10))
    start = pd.Timestamp(end) - pd.DateOffset(years=years) + pd.Timedelta(days=7)
    if cfg.get("start"):
        start = max(start, pd.Timestamp(cfg["start"]))
    return start


def _complete_sessions_only(df, cfg, now_utc):
    """Drop the bar of any session that has not closed (+30 min): a daily bar fetched during
    market hours is partial and must never reach features, signals or the paper ledger."""
    before = len(df)
    df = drop_incomplete_sessions(df, now_utc, cfg.get("calendar") or "NYSE")
    if len(df) < before:
        print(f"[SESSIONS] dropped {before - len(df)} rows from sessions that have not closed yet")
    return df


def load_prices(cfg, symbols, client=None, now_utc=None):
    """Daily bars from Massive only (split-adjusted, adjusted=true). History is the plan's
    rolling window (massive_history_years), never earlier than cfg['start']. Only complete
    sessions are kept."""
    if client is None:
        from etl_common.massive import MassiveClient, load_api_key
        client = MassiveClient(load_api_key())
    now = _utc_now(now_utc)
    end = cfg.get("end") or now.strftime("%Y-%m-%d")
    start = _history_start(cfg, end)
    parts = []
    for s in symbols:
        rows = client.daily_aggs(s, start.strftime("%Y-%m-%d"), end)
        if not rows:
            raise SystemExit(f"Massive returned no daily bars for {s}")
        parts.append(massive_frame(rows, s))
    df = _trading_days(pd.concat(parts, ignore_index=True), cfg.get("calendar"))
    df = _complete_sessions_only(df, cfg, now)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"[MASSIVE] {len(df)} rows {df['date'].min().date()}..{df['date'].max().date()}")
    return df


def load_prices_offline(path, cfg, symbols, now_utc=None):
    """Same frame as load_prices, read from a saved prices_history.csv (no API calls):
    for reproducible local runs and before/after comparisons."""
    now = _utc_now(now_utc)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    df = df[df["ticker"].isin(symbols)]
    end = cfg.get("end") or now.strftime("%Y-%m-%d")
    day = df["date"].dt.tz_localize(None)
    df = df[(day >= _history_start(cfg, end)) & (day <= pd.Timestamp(end))]
    df = _complete_sessions_only(df, cfg, now)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"[OFFLINE] {len(df)} rows {df['date'].min().date()}..{df['date'].max().date()} from {path}")
    return df


def _environment() -> dict:
    """Which platform produced the published numbers: the same config hash on the same prices
    gave DSR 0.817 on Windows / Python 3.13 and 0.577 on the Linux / Python 3.11 CI runner."""
    import numpy, scipy, xgboost, arch, hmmlearn
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "xgboost": xgboost.__version__,
            "arch": arch.__version__, "hmmlearn": hmmlearn.__version__}


def _data_freshness(docs_dir: str, as_of: str, last_session) -> dict:
    """Did the news reach the signal day's cutoff (20:00 UTC) before the features were built?"""
    try:
        with open(os.path.join(docs_dir, "manifest.json"), encoding="utf-8") as fh:
            news = json.load(fh).get("news") or {}
    except (FileNotFoundError, json.JSONDecodeError):
        news = {}
    fetched = news.get("fetched_utc") or news.get("last_updated_utc")
    cutoff = pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(hours=CLOSE_UTC_HOUR)
    return {"prices_as_of": as_of, "last_completed_session": last_session, "news_fetched_utc": fetched,
            "news_cutoff_utc": cutoff.isoformat(),
            "news_fresh": bool(fetched) and pd.Timestamp(fetched) >= cutoff}


def main(cfg, now_utc=None, offline_prices=None, docs_dir=None):
    symbols = universe_symbols(cfg)
    qcfg = cfg.get("quant", {}) or {}
    run_now = _utc_now(now_utc)
    pub = cfg.get("publish", {}) or {}
    docs_dir = docs_dir or pub.get("docs_dir", "docs/data")
    cal = cfg.get("calendar") or "NYSE"

    if offline_prices:
        df = load_prices_offline(offline_prices, cfg, symbols, run_now)
    else:
        df = load_prices(cfg, symbols, now_utc=run_now)

    # Fail loudly instead of publishing empty JSON.
    if df.empty:
        raise SystemExit("Massive returned no data for any symbol")

    # A ticker with zero valid closes must fail the run, never publish a null series.
    no_prices = [s for s in symbols if df.loc[df["ticker"] == s, "close"].notna().sum() == 0]
    if no_prices:
        raise SystemExit(f"Massive returned no valid prices for {no_prices}")

    # === Calidad de datos: lo fatal detiene la corrida; lo demás se marca y se publica ===
    df, quality = validate_prices(df, cal, run_now, qcfg.get("benchmark", "SPY"))
    for f in quality.flags[:10]:
        print(f"[DATA] flag {f}")
    if quality.fatal:
        raise SystemExit("[DATA] fatal data-quality problems: " + " | ".join(quality.fatal))

    # === Dividendos (escala de precios ajustada por splits): largos cobran, cortos pagan ===
    df, div_status = refresh_dividends(df, symbols, os.path.join(docs_dir, "history", "dividends.csv"),
                                       offline=bool(offline_prices))
    print(f"[DIVIDENDS] source={div_status['source']} rows={div_status['rows']} "
          f"ex-dates in prices={int((df['dividends'] > 0).sum())}" +
          (f" error={div_status['error']}" if div_status["error"] else ""))

    # === Resumen para verificación ===
    min_d, max_d = df["date"].min(), df["date"].max()
    print(f"[SUMMARY] prices rows={len(df)} tickers={df['ticker'].nunique()} dates={min_d}..{max_d}")
    for tkr, dfg in df.groupby("ticker"):
        print(f"[SUMMARY] {tkr}: rows={len(dfg)} last={dfg['date'].max()}")

    # === Features (+ noticias point-in-time), régimen, meta-labeling y validación honesta ===
    feats, meta = build_feature_frame(df, qcfg, docs_dir)
    signals, report, internals = make_signals(
        feats, meta, qcfg, cal_name=cfg.get("calendar"),
        trials_path=os.path.join(docs_dir, "history", "trials_log.csv"), return_internals=True,
    )
    as_of = str(df["date"].max().date())
    report.update({"strategy_label": cfg.get("strategy_label"), "data_quality": quality.as_dict(),
                   "dividends": div_status,
                   "data_freshness": _data_freshness(docs_dir, as_of, quality.last_session),
                   "environment": _environment()})
    # Un ticker sin la última sesión publica su señal, pero nunca abre posición ese día.
    signals["data_stale"] = signals["ticker"].isin(quality.stale)
    signals.loc[signals["data_stale"], "kelly_size"] = 0.0
    m = report["meta"]
    print(f"[SUMMARY] config={report['config_hash']} ({cfg.get('strategy_label')}) events={report['n_events']} "
          f"trades={m.get('n_trades')} SR/trade={m.get('sr_per_trade', 0):.3f} DSR={m.get('dsr')} "
          f"trials={report['n_trials']} news_fresh={report['data_freshness']['news_fresh']}")
    print("[SUMMARY] signals\n", signals[["ticker", "regime", "strategy", "side", "p_success", "kelly_size"]])

    # === Guardado ===
    out_dir = cfg.get("output_dir", "data")
    compression = cfg.get("csv_compression", None)  # None | "gzip"
    price_paths = write_single_prices_csv(df, out_dir, compression=compression, now=run_now)
    pred_paths = write_signals_csv(signals, out_dir, now=run_now)

    # === Heartbeat ===
    hb_path = os.path.join(out_dir, "heartbeat.txt")
    with open(hb_path, "w", encoding="utf-8") as hb:
        hb.write(run_now.isoformat())
    print(f"[SUMMARY] heartbeat written at {hb_path}")

    # === Publish to docs/data (private dashboard behind Cloudflare Access) ===
    now = run_now.isoformat()
    window_days = int(pub.get("price_window_days", 60))
    sectors = {t: sec for sec, tickers in (cfg.get("universe") or {}).items() for t in tickers}
    prices_json = write_prices_latest_json(df, os.path.join(docs_dir, "prices_latest.json"), window_days,
                                           now=run_now)
    universe_json = write_json({"sectors": cfg.get("universe") or {"Todos": symbols}, "benchmark": qcfg.get("benchmark")},
                               os.path.join(docs_dir, "universe.json"))
    prices_hist = write_prices_history_csv(df, os.path.join(docs_dir, "history", "prices_history.csv"))
    # === Paper trading (Gate 0): the published signals, executed with next-open fills ===
    ledger, paper = run_paper(df, signals, report, qcfg, os.path.join(docs_dir, "history", "paper_trades.csv"),
                              now=run_now, strategy_label=cfg.get("strategy_label"), cal=cal)
    # === Capa de medición del asesor (informativa: no cambia ninguna operación) ===
    adv = advisor_report.build(df, feats, meta, internals, report, signals, ledger, paper, cfg, docs_dir)
    report["meta"].update(adv["meta"])
    report.update(adv["validation"])
    paper.update(adv["paper"])
    cs = adv["meta"]["calendar"]
    print(f"[ADVISOR] calendar SR={cs.get('sharpe')} sortino={cs.get('sortino')} maxDD={cs.get('max_drawdown')} "
          f"n_eff={adv['meta']['effective_n']:.0f} DSR_eff={adv['meta']['dsr_effective']} "
          f"flags={[f['code'] for f in report['red_flags']]} gate1={paper['gate1']['status']}")
    # Paywall: full signals → premium/, the public file names no pick (publish_json.redact_signals).
    signals_json = write_public_and_premium(signals.assign(sector=signals["ticker"].map(sectors)).to_dict(orient="records"),
                                            docs_dir, "signals_latest.json", redact_signals)
    report_json = write_json({"as_of": as_of, "generated_utc": now, **report},
                             os.path.join(docs_dir, "validation_report.json"))
    risk_json = write_public_and_premium({"generated_utc": now, **adv["risk"]}, docs_dir, "risk_report.json")
    signals_hist = append_signals_history(signals, os.path.join(docs_dir, "history", "signals_history.csv"))
    paper_json = write_public_and_premium(paper, docs_dir, "paper_summary.json", redact_paper)
    print(f"[SUMMARY] paper: {paper['counts']} gate0={paper['gate0']['status']} total_return={paper['total_return']:.4f}")
    update_manifest(os.path.join(docs_dir, "manifest.json"), "paper", {
        "last_updated_utc": now, "as_of": paper["as_of"], "counts": paper["counts"],
        "gate0": paper["gate0"]["status"],
    })

    update_manifest(os.path.join(docs_dir, "manifest.json"), "stocks", {
        "last_updated_utc": now,
        "as_of": as_of,
        "config_hash": report["config_hash"],
        "rows": int(len(df)),
        "tickers": symbols,
        "price_window_days": window_days,
        "dsr": (round(float(m["dsr"]), 4) if pd.notna(m.get("dsr")) else None),
    })
    print(f"[SUMMARY] published: {prices_json}, {signals_json}, {report_json}, {universe_json}, "
          f"{prices_hist}, {signals_hist}")

    return price_paths + pred_paths + [hb_path, prices_json, signals_json, report_json, risk_json, paper_json]


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stocks_etl/config.yaml")
    parser.add_argument("--offline-prices", default=None,
                        help="read prices from a saved prices_history.csv instead of Massive")
    parser.add_argument("--docs-dir", default=None,
                        help="publish into this folder instead of publish.docs_dir (e.g. a temp copy)")
    parser.add_argument("--now", default=None,
                        help="run as if the clock were this UTC time (ISO 8601), for reproducible runs")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = main(cfg, now_utc=args.now, offline_prices=args.offline_prices, docs_dir=args.docs_dir)
    print("Archivos generados:")
    for pth in paths:
        print(" -", pth)

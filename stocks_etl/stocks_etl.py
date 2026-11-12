# stocks_etl/stocks_etl.py — daily ETL; prices from Massive only
import argparse
import os
import warnings
from datetime import datetime, timezone
import pandas as pd
import pandas_market_calendars as mcal

from stocks_etl.features import make_features
from stocks_etl.meta_model import make_signals
from stocks_etl.paper import run_paper
from stocks_etl.publish_json import (
    write_prices_latest_json,
    write_prices_history_csv,
    write_json,
    write_signals_latest_json,
    append_signals_history,
)
from etl_common.manifest import update_manifest

warnings.simplefilter("ignore", FutureWarning)


def write_single_prices_csv(df, base_dir, compression=None):
    """Escribe un SOLO CSV combinado con todos los tickers."""
    if df.empty:
        return []
    os.makedirs(base_dir, exist_ok=True)
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"prices_{date_tag}.csv" + (".gz" if compression == "gzip" else "")
    path = os.path.join(base_dir, filename)
    df.to_csv(path, index=False, compression=compression)
    return [path]


def write_signals_csv(signals, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
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


def load_prices(cfg, symbols, client=None):
    """Daily bars from Massive only (split-adjusted, adjusted=true). History is the plan's
    rolling window (massive_history_years), never earlier than cfg['start']."""
    if client is None:
        from etl_common.massive import MassiveClient, load_api_key
        client = MassiveClient(load_api_key())
    end = cfg.get("end") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    years = int(cfg.get("massive_history_years", 10))
    start = pd.Timestamp(end) - pd.DateOffset(years=years) + pd.Timedelta(days=7)
    if cfg.get("start"):
        start = max(start, pd.Timestamp(cfg["start"]))
    parts = []
    for s in symbols:
        rows = client.daily_aggs(s, start.strftime("%Y-%m-%d"), end)
        if not rows:
            raise SystemExit(f"Massive returned no daily bars for {s}")
        parts.append(massive_frame(rows, s))
    df = _trading_days(pd.concat(parts, ignore_index=True), cfg.get("calendar"))
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"[MASSIVE] {len(df)} rows {df['date'].min().date()}..{df['date'].max().date()}")
    return df




def main(cfg):
    symbols = universe_symbols(cfg)
    qcfg = cfg.get("quant", {}) or {}

    df = load_prices(cfg, symbols)

    # Fail loudly instead of publishing empty JSON.
    if df.empty:
        raise SystemExit("Massive returned no data for any symbol")

    # A ticker with zero valid closes must fail the run, never publish a null series.
    no_prices = [s for s in symbols if df.loc[df["ticker"] == s, "close"].notna().sum() == 0]
    if no_prices:
        raise SystemExit(f"Massive returned no valid prices for {no_prices}")

    # === Resumen para verificación ===
    if not df.empty:
        min_d, max_d = df["date"].min(), df["date"].max()
        print(f"[SUMMARY] prices rows={len(df)} tickers={df['ticker'].nunique()} dates={min_d}..{max_d}")
        for tkr, dfg in df.groupby("ticker"):
            print(f"[SUMMARY] {tkr}: rows={len(dfg)} last={dfg['date'].max()}")

    # === Features, régimen, meta-labeling y validación honesta ===
    feats, meta = make_features(df, qcfg)
    pub = cfg.get("publish", {}) or {}
    docs_dir = pub.get("docs_dir", "docs/data")
    signals, report = make_signals(
        feats, meta, qcfg, cal_name=cfg.get("calendar"),
        trials_path=os.path.join(docs_dir, "history", "trials_log.csv"),
    )
    m = report["meta"]
    print(f"[SUMMARY] events={report['n_events']} trades={m.get('n_trades')} "
          f"SR/trade={m.get('sr_per_trade', 0):.3f} DSR={m.get('dsr')} trials={report['n_trials']}")
    print("[SUMMARY] signals\n", signals[["ticker", "regime", "strategy", "side", "p_success", "kelly_size"]])

    # === Guardado ===
    out_dir = cfg.get("output_dir", "data")
    compression = cfg.get("csv_compression", None)  # None | "gzip"
    price_paths = write_single_prices_csv(df, out_dir, compression=compression)
    pred_paths = write_signals_csv(signals, out_dir)

    # === Heartbeat ===
    hb_path = os.path.join(out_dir, "heartbeat.txt")
    with open(hb_path, "w", encoding="utf-8") as hb:
        hb.write(datetime.now(timezone.utc).isoformat())
    print(f"[SUMMARY] heartbeat written at {hb_path}")

    # === Publish to docs/ (GitHub Pages) ===
    now = datetime.now(timezone.utc).isoformat()
    window_days = int(pub.get("price_window_days", 60))
    sectors = {t: sec for sec, tickers in (cfg.get("universe") or {}).items() for t in tickers}
    prices_json = write_prices_latest_json(df, os.path.join(docs_dir, "prices_latest.json"), window_days)
    signals_json = write_signals_latest_json(signals.assign(sector=signals["ticker"].map(sectors)),
                                             os.path.join(docs_dir, "signals_latest.json"))
    report_json = write_json({"as_of": str(df["date"].max().date()), "generated_utc": now, **report},
                             os.path.join(docs_dir, "validation_report.json"))
    universe_json = write_json({"sectors": cfg.get("universe") or {"Todos": symbols}, "benchmark": qcfg.get("benchmark")},
                               os.path.join(docs_dir, "universe.json"))
    prices_hist = write_prices_history_csv(df, os.path.join(docs_dir, "history", "prices_history.csv"))
    signals_hist = append_signals_history(signals, os.path.join(docs_dir, "history", "signals_history.csv"))
    # === Paper trading (Gate 0): the published signals, executed with next-open fills ===
    ledger, paper = run_paper(df, signals, report, qcfg, os.path.join(docs_dir, "history", "paper_trades.csv"))
    paper_json = write_json(paper, os.path.join(docs_dir, "paper_summary.json"))
    print(f"[SUMMARY] paper: {paper['counts']} gate0={paper['gate0']['status']} total_return={paper['total_return']:.4f}")
    update_manifest(os.path.join(docs_dir, "manifest.json"), "paper", {
        "last_updated_utc": now, "as_of": paper["as_of"], "counts": paper["counts"],
        "gate0": paper["gate0"]["status"],
    })

    update_manifest(os.path.join(docs_dir, "manifest.json"), "stocks", {
        "last_updated_utc": now,
        "as_of": str(df["date"].max().date()),
        "rows": int(len(df)),
        "tickers": symbols,
        "price_window_days": window_days,
        "dsr": (round(float(m["dsr"]), 4) if pd.notna(m.get("dsr")) else None),
    })
    print(f"[SUMMARY] published: {prices_json}, {signals_json}, {report_json}, {universe_json}, "
          f"{prices_hist}, {signals_hist}")

    return price_paths + pred_paths + [hb_path, prices_json, signals_json, report_json, paper_json]


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stocks_etl/config.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = main(cfg)
    print("Archivos generados:")
    for pth in paths:
        print(" -", pth)

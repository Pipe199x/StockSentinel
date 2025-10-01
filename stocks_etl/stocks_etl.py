# stocks_etl/stocks_etl.py
import argparse
import os
import warnings
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

from stocks_etl.features import make_features
from stocks_etl.model_xgb import predict_next_day

warnings.simplefilter("ignore", FutureWarning)


@retry(wait=wait_exponential_jitter(initial=1, max=30), stop=stop_after_attempt(5))
def _download(symbols, start, end, interval, auto_adjust, actions, threads):
    return yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=actions,
        group_by="ticker",
        threads=threads,
        progress=False,
    )


def _normalize(df_raw, symbols, auto_adjust, cal_name):
    """Normaliza a un esquema uniforme y concatena todo en un solo DataFrame."""
    records = []
    cal = mcal.get_calendar(cal_name) if cal_name else None

    for s in symbols:
        dfi = df_raw[s] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw
        if dfi.empty:
            continue

        # reset index y estandarizar nombres
        dfi = dfi.reset_index()
        dfi.columns = [str(c).lower() for c in dfi.columns]

        # detectar columna temporal de forma robusta (date/datetime/primera datetime)
        dt_col = None
        for cand in ("date", "datetime"):
            if cand in dfi.columns:
                dt_col = cand
                break
        if dt_col is None:
            for c in dfi.columns:
                if pd.api.types.is_datetime64_any_dtype(dfi[c]):
                    dt_col = c
                    break
        if dt_col is None and pd.api.types.is_datetime64_any_dtype(dfi.index):
            dfi = dfi.reset_index().rename(columns={"index": "date"})
            dt_col = "date"
        if dt_col is None:
            raise ValueError(f"No se encontró columna de tiempo. Columnas: {list(dfi.columns)}")

        # renombrar OHLC/acciones
        rename = {"adj close": "adj_close", "stock splits": "stock_splits"}
        dfi = dfi.rename(columns=rename)

        # crear columna 'date' UTC y eliminar la original si es distinta
        dfi["date"] = pd.to_datetime(dfi[dt_col], utc=True)
        if dt_col != "date":
            dfi = dfi.drop(columns=[dt_col])

        dfi["ticker"] = s

        # consistencia de columnas
        if auto_adjust and "adj_close" not in dfi.columns:
            dfi["adj_close"] = pd.NA
        for c in ("dividends", "stock_splits"):
            if c not in dfi.columns:
                dfi[c] = 0.0

        # filtrar a días de mercado (seguirá funcionando para diario; en intradía solo valida la fecha)
        if cal is not None and "date" in dfi.columns:
            try:
                start_d = dfi["date"].min().date()
                end_d = dfi["date"].max().date()
                sched = cal.schedule(start_date=start_d, end_date=end_d)
                valid = pd.Index(
                    pd.to_datetime(sched.index)
                    .tz_localize("America/New_York")
                    .tz_convert("UTC")
                    .normalize()
                )
                dfi = dfi[dfi["date"].dt.normalize().isin(valid)]
            except Exception:
                pass

        records.append(
            dfi[
                [
                    "date",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "dividends",
                    "stock_splits",
                ]
            ]
        )

    out = (
        pd.concat(records, ignore_index=True).sort_values(["ticker", "date"])
        if records
        else pd.DataFrame()
    )
    return out


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


def write_predictions_csv(df_preds, base_dir):
    """Escribe el CSV de predicciones (sin la columna 'model')."""
    os.makedirs(base_dir, exist_ok=True)
    df_preds = df_preds.drop(columns=["model"], errors="ignore")
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(base_dir, f"predictions_{date_tag}.csv")
    df_preds.to_csv(path, index=False)
    return [path]


def main(cfg):
    symbols = cfg["symbols"]  # ej: ["AMZN","GOOGL","MSFT"]

    df_raw = _download(
        symbols=symbols,
        start=cfg.get("start"),
        end=cfg.get("end"),
        interval=cfg.get("interval", "1d"),
        auto_adjust=cfg.get("auto_adjust", True),
        actions=cfg.get("actions", True),
        threads=cfg.get("threads", 1),
    )

    df = _normalize(
        df_raw=df_raw,
        symbols=symbols,
        auto_adjust=cfg.get("auto_adjust", True),
        cal_name=cfg.get("calendar"),
    )

    # === Resumen para verificación ===
    if not df.empty:
        min_d, max_d = df["date"].min(), df["date"].max()
        print(f"[SUMMARY] prices rows={len(df)} tickers={df['ticker'].nunique()} dates={min_d}..{max_d}")
        for tkr, dfg in df.groupby("ticker"):
            print(f"[SUMMARY] {tkr}: rows={len(dfg)} last={dfg['date'].max()}")

    # === Features & Predicción (T+1 con XGBoost) ===
    feats = make_features(df)
    preds = predict_next_day(feats, backtest_days=30)
    try:
        print("[SUMMARY] predictions\n", preds[["ticker","as_of","last_close","pred_close_t1"]])
    except Exception:
        print("[SUMMARY] predictions shape:", preds.shape)

    # === Guardado ===
    out_dir = cfg.get("output_dir", "data")
    compression = cfg.get("csv_compression", None)  # None | "gzip"
    price_paths = write_single_prices_csv(df, out_dir, compression=compression)
    pred_paths = write_predictions_csv(preds, out_dir)

    # === Heartbeat ===
    hb_path = os.path.join(out_dir, "heartbeat.txt")
    with open(hb_path, "w", encoding="utf-8") as hb:
        hb.write(datetime.now(timezone.utc).isoformat())
    print(f"[SUMMARY] heartbeat written at {hb_path}")

    return price_paths + pred_paths + [hb_path]


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

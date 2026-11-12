# news_etl/news_client.py
from __future__ import annotations

import logging
from typing import Iterable, List, Dict
from datetime import datetime, timedelta
import argparse
import json
import pathlib

import re


logger = logging.getLogger(__name__)

# Dominios de agregadores/ofertas que generan ruido (no cubren la empresa, solo productos)
EXCLUDE_DOMAINS = [
    "slickdeals.net",
    "dealnews.com",
    "dealcatcher.com",
    "9to5toys.com",
    "techbargains.com",
    "biztoc.com",
    # Plataformas de código/paquetes: lanzamientos de librerías no son noticias
    "pypi.org",
    "github.com",
]

# Señales fuertes de artículos de ofertas/compras; raras en noticias de mercado
DEAL_PATTERN = re.compile(
    r"(% off|free shipping|coupon|promo code|on sale|deal of the day|lowest price"
    r"|prime day|black friday|cyber monday|giveaway|w/ prime)",
    re.IGNORECASE,
)


def is_deal_spam(title: str, summary: str, source: str) -> bool:
    src = (source or "").lower()
    if any(d.split(".")[0] in src for d in EXCLUDE_DOMAINS):
        return True
    return bool(DEAL_PATTERN.search(f"{title or ''} {summary or ''}"))


# -----------------------------
# Utilidades CLI
# -----------------------------
def _ensure_parent(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def _dedupe_by_url(rows: List[dict]) -> List[dict]:
    seen, out = set(), []
    for r in rows:
        u = r.get("url")
        if u and u not in seen:
            out.append(r)
            seen.add(u)
    return out

# -----------------------------
# Massive (/v2/reference/news): tickers come tagged by the provider, no keyword search
# -----------------------------
def massive_rows(articles: Iterable[Dict], ticker: str, universe: set) -> List[dict]:
    """Massive news items -> the RAW row schema used by news_etl (the RAW schema news_etl.news_etl reads)."""
    rows = []
    for a in articles:
        title = a.get("title") or ""
        summary = a.get("description") or ""
        source = (a.get("publisher") or {}).get("name") or "unknown"
        if is_deal_spam(title, summary, source):
            continue
        tagged = [t for t in (a.get("tickers") or []) if t in universe] or [ticker]
        rows.append({
            "published_at": a.get("published_utc"),
            "source": source,
            "title": title,
            "summary": summary,
            "url": a.get("article_url") or "",
            "image_url": a.get("image_url"),
            "raw_tickers": sorted(set(tagged)),
        })
    return rows


def fetch_massive(tickers: List[str], since: datetime, per_day_target: int) -> List[dict]:
    from etl_common.massive import MassiveClient, load_api_key
    client = MassiveClient(load_api_key())
    universe = set(tickers)
    out: List[dict] = []
    for t in tickers:
        rows = massive_rows(client.news(t, since.strftime("%Y-%m-%dT%H:%M:%SZ")), t, universe)
        per_day: Dict[str, int] = {}
        for r in rows:  # newest first: keep the latest per_day_target per (ticker, day)
            day = (r["published_at"] or "")[:10]
            if per_day.get(day, 0) >= per_day_target:
                continue
            per_day[day] = per_day.get(day, 0) + 1
            out.append(r)
        logging.info("[massive] %s: %d articles", t, len(rows))
    return out


# -----------------------------
# CLI para generar NDJSON RAW
# -----------------------------
def run_cli():
    ap = argparse.ArgumentParser(description="Descarga noticias de Massive y genera NDJSON RAW.")
    ap.add_argument("--out", required=True, help="Ruta del NDJSON de salida")
    ap.add_argument("--tickers", required=True, help="Lista separada por coma, ej.: AMZN,MSFT,GOOGL")
    ap.add_argument("--days", type=int, default=2, help="Ventana de días hacia atrás (incluye hoy)")
    ap.add_argument("--per-day-target", type=int, default=20, help="Máx. registros a guardar por (ticker, día) en RAW")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Ventana [hoy-(days-1) .. hoy]
    today_utc = datetime.utcnow().replace(microsecond=0)
    start_window = (today_utc - timedelta(days=max(1, args.days) - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    logging.info("Tickers: %s | Ventana: %s → %s", tickers, start_window.date(), today_utc.date())

    all_rows = _dedupe_by_url(fetch_massive(tickers, start_window, args.per_day_target))
    _ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    if not all_rows:
        logging.warning("No se obtuvieron artículos; el archivo queda vacío.")
    print(f"[news_client] Escrito: {args.out} ({len(all_rows)} líneas, Massive)")


if __name__ == "__main__":
    run_cli()

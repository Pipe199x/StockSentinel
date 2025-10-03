# news_etl/news_client.py
from __future__ import annotations

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
import argparse
import json
import pathlib

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dateutil import parser as dateparser

logger = logging.getLogger(__name__)


@dataclass
class Article:
    published_at: Optional[str]
    source: str
    author: Optional[str]
    title: str
    description: Optional[str]
    url: str
    raw_tickers: List[str]
    content_snippet: Optional[str] = None  # NewsAPI a veces da ~200 chars


class NewsClient(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        from_dt: datetime,
        to_dt: datetime,
        language: Optional[str],
        page_size: int,
        max_pages: int,
    ) -> Iterable[Article]:
        ...


class NewsAPIClient(NewsClient):
    """Cliente NewsAPI (/v2/everything) detrás de una interfaz desacoplada."""
    def __init__(self, api_base: str, api_key_env: str = "NEWSAPI_KEY", request_timeout: float = 20.0):
        self.api_base = api_base.rstrip("/")
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise RuntimeError(f"{api_key_env} no está definido en el entorno.")
        self.timeout = request_timeout
        self._client = httpx.Client(timeout=self.timeout, headers={"X-Api-Key": self.api_key})

    def _extract_tickers(self, text: str) -> List[str]:
        if not text:
            return []
        candidates = {"AMZN", "MSFT", "GOOGL"}
        words = {w.strip("()[],'\".").upper() for w in text.split()}
        return sorted(list(candidates.intersection(words)))

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _request(self, params: Dict) -> Dict:
        r = self._client.get(f"{self.api_base}/everything", params=params)
        r.raise_for_status()
        return r.json()

    def search(
        self,
        query: str,
        from_dt: datetime,
        to_dt: datetime,
        language: Optional[str],
        page_size: int,
        max_pages: int,
    ) -> Iterable[Article]:
        iso_from = from_dt.isoformat(timespec="seconds")
        iso_to = to_dt.isoformat(timespec="seconds")
        page = 1
        while page <= max_pages:
            params = {
                "q": query,
                "from": iso_from,
                "to": iso_to,
                "language": language,
                "pageSize": page_size,
                "page": page,
                "sortBy": "publishedAt",
            }
            data = self._request(params)
            articles = data.get("articles", [])
            if not articles:
                break
            for a in articles:
                try:
                    published_at = dateparser.parse(a.get("publishedAt")).astimezone().isoformat()
                except Exception:
                    published_at = None

                source_name = (a.get("source") or {}).get("name") or "unknown"
                title = a.get("title") or ""
                desc = a.get("description") or ""
                url = a.get("url") or ""
                content = a.get("content") or None  # suele venir truncado

                raw_tickers = sorted(list(set(self._extract_tickers(title) + self._extract_tickers(desc))))
                yield Article(
                    published_at=published_at,
                    source=source_name,
                    author=a.get("author"),
                    title=title,
                    description=desc,
                    url=url,
                    raw_tickers=raw_tickers,
                    content_snippet=content,
                )
            page += 1
            time.sleep(1)

    def close(self):
        self._client.close()


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

def build_query_for_ticker(t: str) -> str:
    t = t.upper().strip()
    if t == "AMZN":
        names = ["Amazon", "AWS", "Prime Video", "Ring", "Alexa"]
    elif t == "MSFT":
        names = ["Microsoft", "Windows", "Azure", "Xbox", "Copilot"]
    elif t == "GOOGL":
        names = ["Google", "Alphabet", "YouTube", "Android", "Gemini"]
    else:
        names = [t]
    # Ej.: (AMZN OR Amazon OR AWS)
    return "(" + " OR ".join([t] + names) + ")"

def _day_bounds(day_utc: datetime) -> (datetime, datetime):
    start = day_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    end = day_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start, end

# -----------------------------
# CLI para generar NDJSON RAW
# -----------------------------
def run_cli():
    ap = argparse.ArgumentParser(description="Descarga noticias con NewsAPI y genera NDJSON RAW (con slicing diario).")
    ap.add_argument("--out", required=True, help="Ruta del NDJSON de salida")
    ap.add_argument("--tickers", required=True, help="Lista separada por coma, ej.: AMZN,MSFT,GOOGL")
    ap.add_argument("--language", default="en")
    ap.add_argument("--days", type=int, default=15, help="Ventana de días hacia atrás (incluye hoy)")
    ap.add_argument("--page-size", type=int, default=25, help="Tamaño de página para NewsAPI")
    ap.add_argument("--max-pages", type=int, default=2, help="Máx. páginas por (ticker, día)")
    ap.add_argument("--per-day-target", type=int, default=20, help="Máx. registros a guardar por (ticker, día) en RAW")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    client = NewsAPIClient(api_base="https://newsapi.org/v2")

    # Ventana [hoy-(days-1) .. hoy]
    today_utc = datetime.utcnow().replace(microsecond=0)
    start_window = (today_utc - timedelta(days=max(1, args.days) - 1)).replace(hour=0, minute=0, second=0, microsecond=0)

    all_rows: List[dict] = []
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    logging.info("Tickers: %s | Ventana diaria: %s → %s", tickers, start_window.date(), today_utc.date())

    try:
        for t in tickers:
            q = build_query_for_ticker(t)
            current = start_window
            while current.date() <= today_utc.date():
                day_from, day_to = _day_bounds(current)
                logging.info("Consultando [%s] %s → %s", t, day_from.date(), day_to.date())

                day_rows: List[dict] = []
                for art in client.search(
                    query=q,
                    from_dt=day_from,
                    to_dt=day_to,
                    language=args.language,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                ):
                    row = {
                        "published_at": art.published_at,
                        "source": art.source,
                        "title": art.title or "",
                        "summary": art.description or art.content_snippet or "",
                        "url": art.url,
                        "raw_tickers": art.raw_tickers,
                    }
                    day_rows.append(row)
                    if len(day_rows) >= args.per_day_target:
                        break

                # Dedupe por día para ese ticker y luego agregar al global
                day_rows = _dedupe_by_url(day_rows)
                all_rows.extend(day_rows)
                current += timedelta(days=1)

    finally:
        client.close()

    all_rows = _dedupe_by_url(all_rows)
    logging.info("Artículos finales (dedupe global): %d", len(all_rows))

    _ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not all_rows:
        logging.warning("No se obtuvieron artículos; el archivo queda vacío.")
    print(f"[news_client] Escrito: {args.out} ({len(all_rows)} líneas)")


if __name__ == "__main__":
    run_cli()

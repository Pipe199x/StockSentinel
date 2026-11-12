"""Minimal Massive (formerly Polygon.io) REST client: live ETL (prices, news) and research.

Reads the key from the MASSIVE_API_KEY environment variable (GitHub Actions secret) or a
local, gitignored .env file. The key is sent only in the Authorization header, never in a
URL or a log line.

License: the Stocks Developer plan is for individual use. Massive data may only be published
to the PRIVATE dashboard (Cloudflare Access) and the private repo, never to a public page.
"""
import os
import time
from pathlib import Path
from typing import Dict, Iterator, Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

BASE_URL = os.getenv("MASSIVE_BASE_URL", "https://api.massive.com")


class RateLimited(Exception):
    pass


def load_api_key(env_file: str = ".env") -> str:
    key = os.getenv("MASSIVE_API_KEY", "")
    if not key and Path(env_file).exists():
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("MASSIVE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("MASSIVE_API_KEY is not set (environment variable or .env)")
    return key


class MassiveClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL, max_per_min: Optional[int] = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.min_interval = 60.0 / max_per_min if max_per_min else 0.0
        self._last = 0.0

    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    @retry(retry=retry_if_exception_type((RateLimited, requests.ConnectionError, requests.Timeout)),
           wait=wait_exponential_jitter(initial=2, max=60), stop=stop_after_attempt(8), reraise=True)
    def get_json(self, url: str, params: Optional[Dict] = None) -> Dict:
        self._throttle()
        if not url.startswith("http"):
            url = self.base_url + url
        r = self.session.get(url, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            raise RateLimited(f"HTTP {r.status_code}")
        if r.status_code in (401, 403):
            raise SystemExit(f"Massive refused the request (HTTP {r.status_code}): "
                             "check the key and that your plan includes this endpoint/history")
        r.raise_for_status()
        return r.json()

    def paginate(self, path: str, params: Optional[Dict] = None) -> Iterator[Dict]:
        data = self.get_json(path, params)
        while True:
            yield from data.get("results", []) or []
            nxt = data.get("next_url")
            if not nxt:
                break
            data = self.get_json(nxt)  # next_url already carries the cursor

    # --- endpoints ---------------------------------------------------------
    def grouped_daily(self, date: str, adjusted: bool = True) -> list:
        """All US stock tickers' OHLCV for one session (T, o, h, l, c, v, vw, t)."""
        data = self.get_json(f"/v2/aggs/grouped/locale/us/market/stocks/{date}",
                             {"adjusted": str(adjusted).lower()})
        return data.get("results", []) or []

    def tickers(self, active: bool, ticker_type: Optional[str] = None) -> Iterator[Dict]:
        params = {"market": "stocks", "active": str(active).lower(), "limit": 1000}
        if ticker_type:
            params["type"] = ticker_type
        return self.paginate("/v3/reference/tickers", params)

    def minute_bars(self, ticker: str, date: str, adjusted: bool = True) -> list:
        data = self.get_json(f"/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}",
                             {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000})
        return data.get("results", []) or []

    def daily_aggs(self, ticker: str, start: str, end: str, adjusted: bool = True) -> list:
        """Daily bars for one ticker (o, h, l, c, v, vw, n, t=ms epoch at the session's start)."""
        data = self.get_json(f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                             {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000})
        return data.get("results", []) or []

    def news(self, ticker: str, since_utc: str, limit: int = 1000) -> Iterator[Dict]:
        """News articles tagged with `ticker`, published at or after `since_utc`, newest first."""
        return self.paginate("/v2/reference/news", {"ticker": ticker, "published_utc.gte": since_utc,
                                                     "order": "desc", "sort": "published_utc", "limit": limit})

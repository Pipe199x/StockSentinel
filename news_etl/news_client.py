from __future__ import annotations

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass

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

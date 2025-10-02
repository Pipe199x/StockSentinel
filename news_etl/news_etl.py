from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Iterable
from urllib.parse import urlparse

# ----- IMPORT corregido para funcionar como paquete o script -----
try:
    # cuando se ejecuta como paquete: python -m news_etl.news_etl
    from .azure_language import AzureLanguageClient  # type: ignore
except ImportError:
    # cuando se ejecuta directamente: python news_etl/news_etl.py
    from azure_language import AzureLanguageClient  # type: ignore
# -----------------------------------------------------------------

import yaml

# -------- logging --------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# -------- utils --------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def today_str() -> str:
    return datetime.now().strftime("%Y%m%d")

def render_output_path(template: str) -> str:
    return template.replace("{{date}}", today_str())

def get_domain(url: str) -> str:
    try:
        netloc = urlparse(url or "").netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""

def matches_any_regex(text: str, patterns: Iterable[str]) -> bool:
    t = text or ""
    for pat in patterns or []:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False

def domain_matches_any(domain: str, patterns: Iterable[str]) -> bool:
    d = (domain or "").lower()
    for p in patterns or []:
        if d == p.lower():
            return True
    return False

def count_regex(pattern: str, text: str) -> int:
    try:
        return len(re.findall(pattern, text or "", flags=re.IGNORECASE))
    except re.error:
        return 0

# -------- heurísticas --------
def is_likely_commerce_post(rec: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    cf = (cfg.get("commerce_filter") or {})
    if not cf.get("enabled", True):
        return False

    title = (rec.get("title") or "")
    desc = (rec.get("description") or "")
    url = (rec.get("url") or "")
    domain = get_domain(url)
    path = urlparse(url).path.lower() if url else ""

    allowed_domains = set((cf.get("allowed_corporate_domains") or []))
    if domain in allowed_domains:
        return False

    title_url_terms = cf.get("title_url_terms") or []
    path_terms = cf.get("path_terms") or []
    term_hit = matches_any_regex(title, title_url_terms) or matches_any_regex(url, title_url_terms)
    path_hit = matches_any_regex(path, path_terms)

    price_rx = cf.get("price_regex") or r"(\$\s?\d+[\d\.,]*)"
    pct_rx = cf.get("percent_regex") or r"\b\d{1,3}\s?%\b"
    min_price = int(cf.get("min_price_tokens", 2))
    min_pct = int(cf.get("min_percent_tokens", 2))
    blob = " ".join([title, desc])
    price_tokens = count_regex(price_rx, blob)
    pct_tokens = count_regex(pct_rx, blob)
    numeric_hit = price_tokens >= min_price or pct_tokens >= min_pct

    return sum([term_hit, path_hit, numeric_hit]) >= 2

def is_registry_release(rec: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    url = rec.get("url") or ""
    domain = get_domain(url)
    path = urlparse(url).path.lower() if url else ""

    if domain in set((cfg.get("registry_domains_blacklist") or [])):
        return True
    if domain == "github.com" and ("/releases" in path or "/tag/" in path):
        return True

    title = (rec.get("title") or "")
    if re.search(r"\b\d+\.\d+(?:\.\d+)?(?:-[A-Za-z0-9\.-]+)?\b", title) and not matches_any_regex(
        title, [r"(?i)amazon", r"(?i)microsoft", r"(?i)google", r"(?i)alphabet"]
    ):
        return True
    return False

# -------- texto para Azure --------
def doc_text(rec: Dict[str, Any]) -> str:
    parts = [rec.get("title") or "", rec.get("description") or "", rec.get("source") or ""]
    return ". ".join([p for p in parts if p]).strip()

# -------- tickers --------
def tag_tickers(rec: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    tt = (cfg.get("ticker_tagging") or {})
    if not tt.get("enabled", True):
        return []
    text = " ".join([
        rec.get("title") or "",
        rec.get("description") or "",
        rec.get("linked_entities_str") or "",
        rec.get("entities_str") or "",
        rec.get("url") or "",
    ])
    hits = []
    for ticker, pats in (tt.get("tickers") or {}).items():
        if matches_any_regex(text, pats):
            hits.append(ticker)
    return sorted(set(hits))

# -------- carga --------
def load_input_ndjson(path: str, sample_limit: int = 0) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for _i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(obj)
            if sample_limit and len(items) >= sample_limit:
                break
    return items

# -------- guardado --------
def save_ndjson(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_csv_preview(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["published_at", "source", "title", "lang", "sentiment_label", "sentiment_score",
            "key_phrases", "entities", "linked_entities", "tickers", "url"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.get("published_at"), r.get("source"), r.get("title"),
                (r.get("language") or {}).get("iso"),
                (r.get("sentiment") or {}).get("label"),
                (r.get("sentiment") or {}).get("score"),
                "; ".join(r.get("key_phrases") or []),
                "; ".join(r.get("entities") or []),
                "; ".join(r.get("linked_entities") or []),
                ",".join(r.get("tickers") or []),
                r.get("url"),
            ])

# -------- main --------
def run(config_path: str = "config.local.yaml") -> None:
    cfg = load_config(config_path)

    input_path = cfg["io"]["input_ndjson"]
    output_ndjson = render_output_path(cfg["io"]["output_ndjson"])
    output_csv = render_output_path(cfg["io"]["output_preview_csv"])
    heartbeat = cfg["io"]["heartbeat_file"]
    sample_limit = int(cfg.get("runtime", {}).get("sample_limit", 0))

    filters_cfg = cfg.get("filters") or {}
    allowed_sources_whitelist = filters_cfg.get("allowed_sources_whitelist") or []
    source_blacklist = filters_cfg.get("source_blacklist") or []
    exclude_phrases = filters_cfg.get("exclude_phrases") or []

    logger.info("Cargando input: %s", input_path)
    raw_items = load_input_ndjson(input_path, sample_limit)

    # dedupe por URL
    seen_urls = set()
    deduped: List[Dict[str, Any]] = []
    for r in raw_items:
        url = r.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        r["source"] = r.get("source") or get_domain(url)
        deduped.append(r)

    logger.info("Entradas: %d | tras dedupe: %d", len(raw_items), len(deduped))

    filtered: List[Dict[str, Any]] = []
    domain_is_commerce: Dict[str, int] = {}
    domain_total: Dict[str, int] = {}

    for rec in deduped:
        domain = get_domain(rec.get("url"))

        if domain_matches_any(domain, source_blacklist):
            continue

        if allowed_sources_whitelist and domain and (domain.lower() not in set(s.lower() for s in allowed_sources_whitelist)):
            continue

        text_for_filter = " ".join([(rec.get("title") or ""), (rec.get("description") or "")])
        if matches_any_regex(text_for_filter, exclude_phrases):
            continue

        if is_registry_release(rec, cfg):
            continue

        commerce = is_likely_commerce_post(rec, cfg)
        if domain:
            domain_total[domain] = domain_total.get(domain, 0) + 1
            if commerce:
                domain_is_commerce[domain] = domain_is_commerce.get(domain, 0) + 1
        if commerce:
            continue

        filtered.append(rec)

    # auto-blacklist por corrida
    ab = cfg.get("auto_blacklist") or {}
    if ab.get("enabled", True):
        min_n = int(ab.get("min_count", 3))
        min_r = float(ab.get("min_ratio", 0.8))
        auto_blocked = {d for d, n in domain_total.items()
                        if n >= min_n and (domain_is_commerce.get(d, 0) / float(n)) >= min_r}
        if auto_blocked:
            logger.info("Auto-blacklist (run): %s", ", ".join(sorted(auto_blocked)))
            filtered = [rec for rec in filtered if get_domain(rec.get("url")) not in auto_blocked]

    logger.info("Tras filtros: %d", len(filtered))

    # Azure Language
    al_cfg = cfg.get("azure_language") or {}
    lang_client = AzureLanguageClient(
        endpoint=al_cfg.get("endpoint", ""),
        key=al_cfg.get("key", ""),
        api_version=al_cfg.get("api_version", "2023-04-01"),
        timeout_seconds=int(al_cfg.get("timeout_seconds", 30)),
    )
    batch_size = int(al_cfg.get("batch_size", 10))
    tasks = al_cfg.get("tasks") or {}

    enriched: List[Dict[str, Any]] = []
    batch: List[Dict[str, Any]] = []

    def flush_batch():
        nonlocal batch
        if not batch:
            return
        try:
            data = lang_client.analyze_batch(
                [{"id": b["id"], "text": b["text"]} for b in batch],
                tasks=tasks,
            )
            parsed = AzureLanguageClient.parse_results(data)
        except Exception as e:
            logger.exception("Azure Language error: %s", e)
            for b in batch:
                rec = b["rec"]
                rec["language"] = {"iso": None, "name": None, "score": 0.0}
                rec["sentiment"] = {"label": None, "score": 0.0}
                rec["key_phrases"] = []
                rec["entities"] = []
                rec["linked_entities"] = []
                rec["tickers"] = []
                enriched.append(rec)
            batch = []
            return

        for b in batch:
            rec = b["rec"]
            rid = b["id"]
            res = parsed.get(rid) or {}
            rec["language"] = res.get("language") or {"iso": None, "name": None, "score": 0.0}
            rec["sentiment"] = res.get("sentiment") or {"label": None, "score": 0.0}
            rec["key_phrases"] = res.get("key_phrases") or []
            rec["entities"] = res.get("entities") or []
            rec["linked_entities"] = res.get("linked_entities") or []
            rec["entities_str"] = "; ".join(rec["entities"])
            rec["linked_entities_str"] = "; ".join(rec["linked_entities"])
            rec["tickers"] = tag_tickers(rec, cfg)
            enriched.append(rec)
        batch = []

    for i, rec in enumerate(filtered, 1):
        text = doc_text(rec)
        batch.append({"id": f"{i}", "text": text, "rec": rec})
        if len(batch) >= batch_size:
            flush_batch()
    flush_batch()

    # guardar
    save_ndjson(output_ndjson, enriched)
    save_csv_preview(output_csv, enriched)

    heartbeat = cfg["io"]["heartbeat_file"]
    os.makedirs(os.path.dirname(heartbeat), exist_ok=True)
    with open(heartbeat, "w", encoding="utf-8") as hb:
        hb.write(datetime.now(timezone.utc).isoformat())

    # stats
    langs, sents = {}, {}
    for r in enriched:
        iso = (r.get("language") or {}).get("iso") or "NA"
        langs[iso] = langs.get(iso, 0) + 1
        label = (r.get("sentiment") or {}).get("label") or "NA"
        sents[label] = sents.get(label, 0) + 1

    logger.info("[SUMMARY] extraidos=%d, escritos=%d", len(deduped), len(enriched))
    logger.info("[SUMMARY] languages=%s", langs)
    logger.info("[SUMMARY] sentiment=%s", sents)
    logger.info("[SUMMARY] outputs: ndjson=%s preview_csv=%s heartbeat=%s",
                output_ndjson, output_csv, heartbeat)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="news_etl/config.local.yaml")
    args = ap.parse_args()
    run(args.config)

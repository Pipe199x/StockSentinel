# news_etl/news_etl.py
import argparse
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any

import requests
import yaml

from news_etl.azure_language import AzureLanguageClient

# --- Azure Blob (opcional)
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except Exception:
    BlobServiceClient = None
    ContentSettings = None

LOG = logging.getLogger(__name__)

# ---------------------------
# Utilidades
# ---------------------------
def render_date_pattern(s: str, date_str: str) -> str:
    return s.replace("{{date}}", date_str)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        txt = os.path.expandvars(f.read())
    return yaml.safe_load(txt)

def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def read_ndjson(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_ndjson(path: str, rows: List[dict]):
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv_preview(path: str, rows: List[dict], cols: List[str]):
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                s = str(v).replace('"', '""')
                vals.append(f"\"{s}\"")
            f.write(",".join(vals) + "\n")

def domain_from_url(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", re.findall(r"https?://([^/]+)", url, re.I)[0].lower())
    except Exception:
        return ""

# ---------------------------
# Filtros
# ---------------------------
def passes_filters(item: dict, cfg: dict) -> bool:
    src = (item.get("source") or "").lower()
    title = item.get("title") or ""
    url = item.get("url") or ""
    dom = domain_from_url(url)

    fcfg = cfg.get("filters", {})
    wl = [d.lower() for d in fcfg.get("allowed_sources_whitelist", [])]
    bl = [d.lower() for d in fcfg.get("source_blacklist", [])]
    if wl and src not in wl:
        return False
    if src in bl or dom in bl:
        return False

    for rx in fcfg.get("exclude_phrases", []):
        try:
            if re.search(rx, title) or re.search(rx, url):
                return False
        except re.error:
            # Si hay una regex inválida, la ignoramos
            pass

    # Registry blacklist
    if dom in [d.lower() for d in cfg.get("registry_domains_blacklist", [])]:
        return False

    # Commerce filter
    c = cfg.get("commerce_filter", {})
    if c.get("enabled", False):
        allowed = [d.lower() for d in c.get("allowed_corporate_domains", [])]
        for rx in c.get("title_url_terms", []):
            if re.search(rx, title) or re.search(rx, url):
                if dom not in allowed:
                    return False
        for rx in c.get("path_terms", []):
            if re.search(rx, url):
                if dom not in allowed:
                    return False
        # Señales fuertes de precio y %
        try:
            price_rx = re.compile(c["price_percent_rules"]["price_regex"])
            perc_rx = re.compile(c["price_percent_rules"]["percent_regex"])
            price_hits = len(price_rx.findall(title + " " + url))
            perc_hits = len(perc_rx.findall(title + " " + url))
            if price_hits >= c["price_percent_rules"]["min_price_tokens"] or \
               perc_hits >= c["price_percent_rules"]["min_percent_tokens"]:
                if dom not in allowed:
                    return False
        except Exception:
            pass

    return True

def dedupe_by_url(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for it in items:
        u = it.get("url")
        if u and u not in seen:
            out.append(it)
            seen.add(u)
    return out

# ---------------------------
# Ticker tagging (usa texto + entidades)
# ---------------------------
def tag_tickers(item: dict, cfg: dict) -> List[str]:
    if not cfg.get("ticker_tagging", {}).get("enabled", False):
        return []
    # Texto base
    text = " ".join([
        item.get("title", "") or "",
        item.get("summary", "") or "",
        item.get("source", "") or "",
    ])

    # Añadimos entidades detectadas (reconocidas + enlazadas)
    ents = item.get("entities") or []
    lents = item.get("linked_entities") or []
    if isinstance(ents, list):
        text += " " + " ".join(ents)
    if isinstance(lents, list):
        text += " " + " ".join(lents)

    out = []
    for ticker, patterns in (cfg.get("ticker_tagging", {}).get("tickers", {}) or {}).items():
        for rx in patterns:
            try:
                if re.search(rx, text):
                    out.append(ticker)
                    break
            except re.error:
                # si la regex estuviera mal, la ignoramos
                continue
    return sorted(list(set(out)))

# ---------------------------
# Azure Blob helpers
# ---------------------------
def _blob_container_client_from_cfg(blob_cfg: dict):
    if not blob_cfg.get("enabled", False):
        return None
    if BlobServiceClient is None:
        LOG.warning("azure-storage-blob no instalado; omitiendo upload")
        return None
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or blob_cfg.get("connection_string")
    if not conn:
        LOG.warning("Connection String vacío; omitiendo upload")
        return None
    svc = BlobServiceClient.from_connection_string(conn)
    return svc.get_container_client(blob_cfg["container"])

def _content_type(path: str, blob_cfg: dict):
    if ContentSettings is None:
        return None
    ext = os.path.splitext(path.lower())[1]
    ct = (blob_cfg.get("content_types") or {}).get(ext)
    return ContentSettings(content_type=ct) if ct else None

def upload_files_to_blob(cfg: dict, local_paths: Dict[str, str]):
    blob_cfg = cfg.get("azure_blob", {})
    if not blob_cfg.get("enabled", False):
        return
    cont = _blob_container_client_from_cfg(blob_cfg)
    if cont is None:
        return
    prefix = blob_cfg.get("prefix", "").strip("/")
    for name, local_path in local_paths.items():
        remote = f"{prefix}/{os.path.basename(local_path)}" if prefix else os.path.basename(local_path)
        LOG.info("[BLOB] Subiendo %s -> %s", local_path, remote)
        with open(local_path, "rb") as f:
            cont.upload_blob(
                name=remote,
                data=f,
                overwrite=True,
                content_settings=_content_type(local_path, blob_cfg),
            )

# ---------------------------
# ETL principal
# ---------------------------
def run(config_path: str):
    cfg = load_yaml(config_path)
    log_level = getattr(logging, (cfg.get("runtime", {}).get("log_level") or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    date_str = datetime.now().strftime("%Y%m%d")
    io_cfg = cfg["io"]

    input_path = io_cfg["input_ndjson"]
    output_ndjson = render_date_pattern(io_cfg["output_ndjson"], date_str)
    output_preview = render_date_pattern(io_cfg["output_preview_csv"], date_str)
    heartbeat = io_cfg["heartbeat_file"]

    LOG.info("Cargando input: %s", input_path)
    items = read_ndjson(input_path)
    items = dedupe_by_url(items)
    LOG.info("Entradas: %d | tras dedupe: %d", len(items), len(items))

    # Muestreo opcional
    try:
        sample_limit = int(str(cfg.get("runtime", {}).get("sample_limit", "0")))
    except Exception:
        sample_limit = 0
    if sample_limit and len(items) > sample_limit:
        items = items[:sample_limit]

    # Filtros
    filtered = [it for it in items if passes_filters(it, cfg)]
    LOG.info("Tras filtros: %d", len(filtered))

    # --- Azure Language
    lang_cfg = cfg["azure_language"]
    lang_client = AzureLanguageClient(
        endpoint=os.path.expandvars(lang_cfg["endpoint"]),
        key=os.path.expandvars(lang_cfg["key"]),
        api_version=lang_cfg["api_version"],
        force_language=(lang_cfg.get("force_language") or "").strip() or None,
        timeout_seconds=int(lang_cfg.get("timeout_seconds", 30)),
    )
    batch_size = max(1, min(5, int(lang_cfg.get("batch_size", 5))))  # API limita a 5
    task_flags = (lang_cfg.get("tasks") or {
        "language_detection": True,
        "sentiment_analysis": True,
        "key_phrase_extraction": True,
        "entity_recognition": True,
        "entity_linking": True,
    })

    enriched: List[dict] = []

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    for chunk in chunks(filtered, batch_size):
        # ids locales en el batch
        docs = [{
            "id": str(i + 1),
            "text": f"{it.get('title','')}. {it.get('summary','') or ''} {it.get('source','') or ''}".strip()
        } for i, it in enumerate(chunk)]

        try:
            combined = lang_client.analyze_batch(docs, tasks=task_flags)
            parsed = AzureLanguageClient.parse_results(combined)

            for i, it in enumerate(chunk):
                rid = str(i + 1)
                res = parsed.get(rid, {})
                it_out = dict(it)
                it_out["language"] = res.get("language")
                it_out["sentiment"] = res.get("sentiment")
                it_out["key_phrases"] = res.get("key_phrases", [])
                it_out["entities"] = res.get("entities", [])
                it_out["linked_entities"] = res.get("linked_entities", [])
                it_out["tickers"] = tag_tickers(it_out, cfg)
                enriched.append(it_out)

        except requests.HTTPError as e:
            LOG.error("Azure Language error: %s", e)
        except Exception as e:
            LOG.exception("Error procesando batch: %s", e)

    # Salidas locales
    write_ndjson(output_ndjson, enriched)

    preview_cols = [
        "published_at", "source", "title",
        "lang", "sentiment_label", "sentiment_score",
        "tickers", "url"
    ]
    preview_rows = []
    for r in enriched:
        lang = r.get("language") or {}
        lang_val = lang.get("iso") or lang.get("name") or ""
        prev = {
            "published_at": r.get("published_at", ""),
            "source": r.get("source", ""),
            "title": r.get("title", ""),
            "lang": lang_val,
            "sentiment_label": (r.get("sentiment") or {}).get("label"),
            "sentiment_score": (r.get("sentiment") or {}).get("score", 0.0),
            "tickers": ",".join(r.get("tickers", [])),
            "url": r.get("url", ""),
        }
        preview_rows.append(prev)
    write_csv_preview(output_preview, preview_rows, preview_cols)

    ensure_dir_for_file(heartbeat)
    with open(heartbeat, "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat())

    # Subida a Blob (datasets/news/…)
    upload_files_to_blob(cfg, {
        "ndjson": output_ndjson,
        "preview_csv": output_preview,
        "heartbeat": heartbeat,
    })

    # Resumen
    lang_counts = Counter([ (r.get("language") or {}).get("iso") or (r.get("language") or {}).get("name") for r in enriched]) or {"NA": len(enriched)}
    sent_counts = Counter([(r.get("sentiment") or {}).get("label") for r in enriched]) or {"NA": len(enriched)}
    LOG.info("[SUMMARY] extraidos=%d, escritos=%d", len(items), len(enriched))
    LOG.info("[SUMMARY] languages=%s", dict(lang_counts))
    LOG.info("[SUMMARY] sentiment=%s", dict(sent_counts))
    LOG.info("[SUMMARY] outputs: ndjson=%s preview_csv=%s heartbeat=%s", output_ndjson, output_preview, heartbeat)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)

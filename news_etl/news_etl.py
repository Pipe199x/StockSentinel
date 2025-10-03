# news_etl/news_etl.py
import argparse
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Any

import yaml
import requests

from news_etl.azure_language import AzureLanguageClient

# --- Azure Blob (opcional)
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except Exception:
    BlobServiceClient = None
    ContentSettings = None

LOG = logging.getLogger(__name__)

# =========================
# Utilidades de archivo
# =========================
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
    items: List[dict] = []
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

# =========================
# Filtros
# =========================
def domain_from_url(url: str) -> str:
    try:
        m = re.findall(r"https?://([^/]+)", url, re.I)
        if not m:
            return ""
        dom = m[0].lower()
        return re.sub(r"^www\.", "", dom)
    except Exception:
        return ""

def passes_filters(item: dict, cfg: dict) -> bool:
    src = (item.get("source") or "").lower()
    title = item.get("title") or ""
    url = item.get("url") or ""
    dom = domain_from_url(url)

    fcfg = cfg.get("filters", {}) or {}

    # Whitelist
    wl = [d.lower() for d in fcfg.get("allowed_sources_whitelist", [])]
    if wl and src not in wl:
        return False

    # Blacklist por source/dom
    bl = [d.lower() for d in fcfg.get("source_blacklist", [])]
    if src in bl or dom in bl:
        return False

    # Regex de exclusión por título/URL
    for rx in fcfg.get("exclude_phrases", []):
        if re.search(rx, title, re.I) or re.search(rx, url, re.I):
            return False

    # Blacklist de registro
    if dom in [d.lower() for d in cfg.get("registry_domains_blacklist", [])]:
        return False

    # Commerce filter
    c = cfg.get("commerce_filter", {}) or {}
    if c.get("enabled"):
        allowed = [d.lower() for d in c.get("allowed_corporate_domains", [])]

        for rx in c.get("title_url_terms", []):
            if re.search(rx, title, re.I) or re.search(rx, url, re.I):
                if dom not in allowed:
                    return False
        for rx in c.get("path_terms", []):
            if re.search(rx, url, re.I):
                if dom not in allowed:
                    return False

        # Señales fuertes de precio/%
        prx = re.compile(c["price_percent_rules"]["price_regex"])
        prc = re.compile(c["price_percent_rules"]["percent_regex"])
        price_hits = len(prx.findall(f"{title} {url}"))
        perc_hits = len(prc.findall(f"{title} {url}"))
        if price_hits >= c["price_percent_rules"]["min_price_tokens"] or \
           perc_hits  >= c["price_percent_rules"]["min_percent_tokens"]:
            if dom not in allowed:
                return False

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

# =========================
# Tagging por ticker
# =========================
def tag_tickers(item: dict, cfg: dict) -> List[str]:
    if not cfg.get("ticker_tagging", {}).get("enabled", False):
        return []

    patterns_by_ticker = cfg["ticker_tagging"]["tickers"]
    text = " ".join([
        item.get("title","") or "",
        item.get("summary","") or "",
        item.get("source","") or "",
    ])

    # también usamos entidades devueltas por Azure
    entities = item.get("entities") or []
    linked = item.get("linked_entities") or []
    entity_pool = set([e.strip() for e in entities + linked if isinstance(e, str)])

    found = set()

    for ticker, patterns in patterns_by_ticker.items():
        hit = False
        for rx in patterns:
            if re.search(rx, text):
                hit = True
                break
            # match directo contra entidades
            for name in entity_pool:
                if re.search(rx, name):
                    hit = True
                    break
            if hit:
                break
        if hit:
            found.add(ticker)

    return sorted(list(found))

# =========================
# Límite por (día, ticker) SOLO en scope
# =========================
def _date_key(dt_str: str) -> str:
    if not dt_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return "unknown"

def apply_scoped_limit(enriched_rows: List[dict], cfg: dict) -> List[dict]:
    rt = cfg.get("runtime", {}) or {}
    N = int(rt.get("limit_per_ticker_per_day", 0) or 0)
    scope = set([t.upper() for t in (rt.get("limit_ticker_scope") or [])])
    if N <= 0 or not scope:
        return enriched_rows

    rows = list(enriched_rows)
    if bool(rt.get("newest_first", True)):
        rows.sort(key=lambda r: (r.get("published_at") or ""), reverse=True)

    counters: Dict[tuple, int] = {}
    kept_ids = set()
    for r in rows:
        day = _date_key(r.get("published_at") or "")
        if day == "unknown":
            kept_ids.add(id(r))
            continue

        tickers = [t.upper() for t in (r.get("tickers") or [])]
        in_scope = [t for t in tickers if t in scope]
        if not in_scope:
            kept_ids.add(id(r))
            continue

        # elegir ticker del scope con menor consumo ese día
        best_t = None
        best_val = None
        for t in in_scope:
            v = counters.get((day, t), 0)
            if best_val is None or v < best_val:
                best_val = v
                best_t = t

        if best_t is None:
            kept_ids.add(id(r))
            continue

        if counters.get((day, best_t), 0) < N:
            kept_ids.add(id(r))
            counters[(day, best_t)] = counters.get((day, best_t), 0) + 1
        # else: descartada por límite

    # volver al orden original de enriched_rows
    final = [r for r in enriched_rows if id(r) in kept_ids]
    return final

# =========================
# Azure Blob helpers
# =========================
def _blob_container_client_from_cfg(blob_cfg: dict):
    if not blob_cfg.get("enabled", False):
        return None
    if BlobServiceClient is None:
        LOG.warning("azure-storage-blob no instalado; omitiendo upload.")
        return None
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or blob_cfg.get("connection_string")
    if not conn:
        LOG.warning("Connection String vacío; omitiendo upload.")
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
    blob_cfg = cfg.get("azure_blob", {}) or {}
    if not blob_cfg.get("enabled", False):
        return
    cont = _blob_container_client_from_cfg(blob_cfg)
    if cont is None:
        return
    prefix = blob_cfg.get("prefix", "").strip("/")
    for _, local_path in local_paths.items():
        remote = f"{prefix}/{os.path.basename(local_path)}" if prefix else os.path.basename(local_path)
        LOG.info("[BLOB] Subiendo %s -> %s", local_path, remote)
        with open(local_path, "rb") as f:
            cont.upload_blob(
                name=remote,
                data=f,
                overwrite=True,
                content_settings=_content_type(local_path, blob_cfg),
            )

# =========================
# ETL principal
# =========================
def run(config_path: str):
    cfg = load_yaml(config_path)
    log_level = getattr(logging, (cfg.get("runtime", {}).get("log_level") or "INFO").upper())
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    io_cfg = cfg["io"]

    input_path = io_cfg["input_ndjson"]
    output_ndjson = render_date_pattern(io_cfg["output_ndjson"], date_str)
    output_preview = render_date_pattern(io_cfg["output_preview_csv"], date_str)
    heartbeat = io_cfg["heartbeat_file"]

    LOG.info("Cargando input: %s", input_path)
    items = read_ndjson(input_path)
    items = dedupe_by_url(items)
    LOG.info("Entradas: %d | tras dedupe: %d", len(items), len(items))

    # Filtros
    filtered = [it for it in items if passes_filters(it, cfg)]
    LOG.info("Tras filtros: %d", len(filtered))

    # Azure Language
    lang_cfg = cfg["azure_language"]
    flags = lang_cfg.get("tasks", {}) or {}
    lang_client = AzureLanguageClient(
        endpoint=os.path.expandvars(lang_cfg["endpoint"]),
        key=os.path.expandvars(lang_cfg["key"]),
        api_version=lang_cfg.get("api_version", "2023-04-01"),
        force_language=(lang_cfg.get("force_language") or "").strip() or None,
        timeout_seconds=int(lang_cfg.get("timeout_seconds", 30)),
    )
    batch_size = max(1, min(5, int(lang_cfg.get("batch_size", 5))))

    def batched(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    enriched: List[dict] = []

    for chunk in batched(filtered, batch_size):
        # importante: ids 1..len(chunk) por batch (parser usa string id)
        docs = []
        for idx, it in enumerate(chunk, start=1):
            text = " ".join([
                (it.get("title") or "").strip(),
                (it.get("summary") or "").strip(),
                (it.get("source") or "").strip(),
            ]).strip()
            docs.append({"id": str(idx), "text": text})

        try:
            grouped = lang_client.analyze_batch(docs, tasks=flags)
            per_doc = AzureLanguageClient.parse_results(grouped)

            # fusionar resultados por posición del chunk
            for idx, it in enumerate(chunk, start=1):
                rid = str(idx)
                res = per_doc.get(rid, {}) or {}
                out = dict(it)
                out["language"] = res.get("language")
                out["sentiment"] = res.get("sentiment")
                # Si quisieras agregar scores detallados, podrías extender parse_results
                out["key_phrases"] = res.get("key_phrases", [])
                out["entities"] = res.get("entities", [])
                out["linked_entities"] = res.get("linked_entities", [])
                # Tagging final basado en texto + entidades
                out["tickers"] = tag_tickers(out, cfg)
                enriched.append(out)

        except requests.HTTPError as e:
            LOG.error("Azure Language error: %s", e)
        except Exception as e:
            LOG.exception("Error procesando batch: %s", e)

    # === Límite SOLO para tickers en scope (conserva el resto)
    enriched_limited = apply_scoped_limit(enriched, cfg)

    # Escribir NDJSON
    write_ndjson(output_ndjson, enriched_limited)

    # Construir preview CSV (con columnas extra)
    preview_rows = []
    for r in enriched_limited:
        prev = {
            "published_at": r.get("published_at",""),
            "source": r.get("source",""),
            "title": r.get("title",""),
            "lang": (r.get("language") or {}).get("iso") if isinstance(r.get("language"), dict) else (r.get("language") or ""),
            "sentiment_label": (r.get("sentiment") or {}).get("label"),
            "sentiment_score": (r.get("sentiment") or {}).get("score", 0.0),
            "tickers": ",".join(r.get("tickers", [])),
            "url": r.get("url",""),
            "key_phrases": "|".join(r.get("key_phrases", [])),
            "entities": "|".join(r.get("entities", [])),
            "linked_entities": "|".join(r.get("linked_entities", [])),
        }
        preview_rows.append(prev)

    preview_cols = [
        "published_at","source","title","lang",
        "sentiment_label","sentiment_score",
        "tickers","url",
        "key_phrases","entities","linked_entities",
    ]
    write_csv_preview(output_preview, preview_rows, preview_cols)

    # Heartbeat
    ensure_dir_for_file(heartbeat)
    with open(heartbeat, "w", encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).isoformat())

    # Subida a Blob
    upload_files_to_blob(cfg, {
        "ndjson": output_ndjson,
        "preview_csv": output_preview,
        "heartbeat": heartbeat,
    })

    # Resumen
    lang_counts = Counter([ (r.get("language") or {}).get("iso") if isinstance(r.get("language"), dict) else r.get("language")
                           for r in enriched_limited]) or {"NA": len(enriched_limited)}
    sent_counts = Counter([(r.get("sentiment") or {}).get("label") for r in enriched_limited]) or {"NA": len(enriched_limited)}
    LOG.info("[SUMMARY] extraidos=%d, escritos=%d", len(items), len(enriched_limited))
    LOG.info("[SUMMARY] languages=%s", dict(lang_counts))
    LOG.info("[SUMMARY] sentiment=%s", dict(sent_counts))
    LOG.info("[SUMMARY] outputs: ndjson=%s preview_csv=%s heartbeat=%s", output_ndjson, output_preview, heartbeat)

# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)

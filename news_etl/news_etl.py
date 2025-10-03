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
except Exception:  # paquete no instalado en local
    BlobServiceClient = None
    ContentSettings = None

LOG = logging.getLogger(__name__)

# =============================================================================
# Utilidades básicas
# =============================================================================
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


def _stringify_cell(v: Any) -> str:
    """
    Normaliza valores para CSV: listas -> '; ' join, dict -> JSON, resto -> str.
    """
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if v is None:
        return ""
    return str(v)


def write_csv_preview(path: str, rows: List[dict], cols: List[str]):
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                s = _stringify_cell(r.get(c, ""))
                # escapa comillas
                s = s.replace('"', '""')
                vals.append(f"\"{s}\"")
            f.write(",".join(vals) + "\n")


def domain_from_url(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", re.findall(r"https?://([^/]+)", url, re.I)[0].lower())
    except Exception:
        return ""

# =============================================================================
# Filtros
# =============================================================================
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
        if re.search(rx, title) or re.search(rx, url):
            return False

    # blacklist de registro
    if dom in [d.lower() for d in cfg.get("registry_domains_blacklist", [])]:
        return False

    # filtro commerce
    c = cfg.get("commerce_filter", {})
    if c.get("enabled", False):
        for rx in c.get("title_url_terms", []):
            if re.search(rx, title) or re.search(rx, url):
                allowed = [d.lower() for d in c.get("allowed_corporate_domains", [])]
                if dom not in allowed:
                    return False
        for rx in c.get("path_terms", []):
            if re.search(rx, url):
                allowed = [d.lower() for d in c.get("allowed_corporate_domains", [])]
                if dom not in allowed:
                    return False

        price_rx = re.compile(c["price_percent_rules"]["price_regex"])
        perc_rx = re.compile(c["price_percent_rules"]["percent_regex"])
        tokens = title + " " + url
        price_hits = len(price_rx.findall(tokens))
        perc_hits = len(perc_rx.findall(tokens))
        if price_hits >= c["price_percent_rules"]["min_price_tokens"] or \
           perc_hits >= c["price_percent_rules"]["min_percent_tokens"]:
            allowed = [d.lower() for d in c.get("allowed_corporate_domains", [])]
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

# =============================================================================
# Tagging de tickers (regex + entidades)
# =============================================================================
def tag_tickers(item: dict, cfg: dict, ent_names: List[str], linked_names: List[str]) -> List[str]:
    tick_cfg = cfg.get("ticker_tagging", {})
    if not tick_cfg.get("enabled", False):
        return []

    text = f"{item.get('title','')} {item.get('summary','')} {item.get('source','')}"
    text_lower = text.lower()
    ents_lower = {e.lower() for e in ent_names + linked_names}

    out = set()

    # reglas explícitas por regex
    for ticker, patterns in tick_cfg.get("tickers", {}).items():
        for rx in patterns:
            if re.search(rx, text, re.I):
                out.add(ticker)
                break

    # mapping por entidades (si aparecen en Entities/LinkedEntities)
    ent_map = tick_cfg.get("entity_name_map", {})  # {"microsoft": "MSFT", ...}
    for name, tk in ent_map.items():
        if (name.lower() in text_lower) or (name.lower() in ents_lower):
            out.add(tk)

    return sorted(out)

# =============================================================================
# Azure Blob helpers (Connection String)
# =============================================================================
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
    for _, local_path in local_paths.items():
        remote_name = os.path.basename(local_path)
        remote = f"{prefix}/{remote_name}" if prefix else remote_name
        LOG.info("[BLOB] Subiendo %s -> %s", local_path, remote)
        with open(local_path, "rb") as f:
            cont.upload_blob(
                name=remote,
                data=f,
                overwrite=True,
                content_settings=_content_type(local_path, blob_cfg),
            )

# =============================================================================
# Límite por ticker / día
# =============================================================================
def _date_utc(iso_str: str) -> str:
    """YYYY-MM-DD en UTC."""
    if not iso_str:
        return "1970-01-01"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.date().isoformat()
    except Exception:
        return iso_str[:10]


def limit_per_ticker_per_day(items: List[dict], limit: int, scope: set[str]) -> List[dict]:
    """
    Mantiene como máximo `limit` items por (ticker, día) en UTC.
    Una nota con múltiples tickers se acepta si alguno tiene cupo; al aceptar, consume
    cupo de todos los tickers involucrados (intersección con scope).
    Orden: fecha desc + score de sentimiento desc.
    """
    if limit <= 0 or not scope:
        return items

    def _key(r):
        s = (r.get("sentiment") or {}).get("score") or 0.0
        return (_date_utc(r.get("published_at", "")), s)

    ordered = sorted(items, key=_key, reverse=True)
    used: dict[tuple[str, str], int] = {}
    out: List[dict] = []

    for r in ordered:
        tickers = [t for t in r.get("tickers", []) if t in scope]
        if not tickers:
            continue
        day = _date_utc(r.get("published_at", ""))
        if not any(used.get((t, day), 0) < limit for t in tickers):
            continue
        # acepta
        for t in tickers:
            k = (t, day)
            used[k] = used.get(k, 0) + 1
        out.append(r)

    return out

# =============================================================================
# ETL principal
# =============================================================================
def run(config_path: str):
    cfg = load_yaml(config_path)
    log_level = getattr(logging, (cfg.get("runtime", {}).get("log_level") or "INFO").upper())
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    date_str = datetime.utcnow().strftime("%Y%m%d")
    io_cfg = cfg["io"]

    input_path = io_cfg["input_ndjson"]
    output_ndjson = render_date_pattern(io_cfg["output_ndjson"], date_str)
    output_preview = render_date_pattern(io_cfg["output_preview_csv"], date_str)
    heartbeat = io_cfg["heartbeat_file"]

    LOG.info("Cargando input: %s", input_path)
    items = read_ndjson(input_path)
    items = dedupe_by_url(items)
    LOG.info("Entradas: %d | tras dedupe: %d", len(items), len(items))

    # muestreo (debug)
    try:
        sample_limit = int(str(cfg.get("runtime", {}).get("sample_limit", "0")))
    except Exception:
        sample_limit = 0
    if sample_limit and len(items) > sample_limit:
        items = items[:sample_limit]

    # filtros
    filtered = [it for it in items if passes_filters(it, cfg)]
    LOG.info("Tras filtros: %d", len(filtered))

    # ---------------- Azure Language ----------------
    lang_cfg = cfg["azure_language"]
    lang_client = AzureLanguageClient(
        endpoint=os.path.expandvars(lang_cfg["endpoint"]),
        key=os.path.expandvars(lang_cfg["key"]),
        api_version=lang_cfg.get("api_version", "2023-04-01"),
        force_language=(lang_cfg.get("force_language") or "").strip() or None,
        timeout_seconds=int(lang_cfg.get("timeout_seconds", 30)),
    )
    batch_size = max(1, min(5, int(lang_cfg.get("batch_size", 5))))  # API limita a 5
    tasks_flags = lang_cfg.get("tasks", {
        "language_detection": True,
        "sentiment_analysis": True,
        "key_phrase_extraction": True,
        "entity_recognition": True,
        "entity_linking": True,
    })

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    enriched: List[dict] = []

    for chunk in _chunks(filtered, batch_size):
        docs = [
            {"id": str(i + 1),
             "text": f"{it.get('title','')}. {it.get('summary','') or ''} {it.get('source','') or ''}".strip()}
            for i, it in enumerate(chunk)
        ]
        try:
            grouped = lang_client.analyze_batch(docs, tasks=tasks_flags)
            per_doc = AzureLanguageClient.parse_results(grouped)

            for i, it in enumerate(chunk):
                rid = str(i + 1)
                res = per_doc.get(rid, {})
                it_out = dict(it)
                # enriquecidos
                lang_obj = res.get("language")
                it_out["language"] = lang_obj
                it_out["sentiment"] = res.get("sentiment", {"label": None, "score": 0.0})
                it_out["key_phrases"] = res.get("key_phrases", [])
                ent = res.get("entities", [])
                link_ent = res.get("linked_entities", [])
                it_out["entities"] = ent
                it_out["linked_entities"] = link_ent
                # tagging de tickers (regex + entidades)
                it_out["tickers"] = tag_tickers(it_out, cfg, ent, link_ent)
                enriched.append(it_out)

        except requests.HTTPError as e:
            LOG.error("Azure Language error: %s", e)
        except Exception as e:
            LOG.exception("Error procesando batch: %s", e)

    # ------------- límite por ticker/día (opcional) -------------
    rt = cfg.get("runtime", {})
    per_day_limit = int(rt.get("limit_per_ticker_per_day", 0))
    scope = set(rt.get("limit_ticker_scope") or [])
    if per_day_limit > 0 and scope:
        before = len(enriched)
        enriched = limit_per_ticker_per_day(enriched, per_day_limit, scope)
        LOG.info("Aplicado límite %d por ticker/día en %s: %d -> %d filas",
                 per_day_limit, sorted(scope), before, len(enriched))

    # ---------------- Outputs locales ----------------
    write_ndjson(output_ndjson, enriched)

    preview_rows = []
    for r in enriched:
        prev = {
            "published_at": r.get("published_at", ""),
            "source": r.get("source", ""),
            "title": r.get("title", ""),
            "lang": (r.get("language") or {}).get("iso") or "",
            "sentiment_label": (r.get("sentiment") or {}).get("label"),
            "sentiment_score": (r.get("sentiment") or {}).get("score", 0.0),
            "tickers": r.get("tickers", []),
            "url": r.get("url", ""),
            "key_phrases": r.get("key_phrases", []),
            "entities": r.get("entities", []),
            "linked_entities": r.get("linked_entities", []),
        }
        preview_rows.append(prev)

    preview_cols = [
        "published_at", "source", "title", "lang",
        "sentiment_label", "sentiment_score",
        "tickers", "url",
        "key_phrases", "entities", "linked_entities",
    ]
    write_csv_preview(output_preview, preview_rows, preview_cols)

    ensure_dir_for_file(heartbeat)
    with open(heartbeat, "w", encoding="utf-8") as f:
        f.write(datetime.utcnow().isoformat())

    # ---------------- Subida a Blob ----------------
    upload_files_to_blob(cfg, {
        "ndjson": output_ndjson,
        "preview_csv": output_preview,
        "heartbeat": heartbeat,
    })

    # ---------------- Resumen ----------------
    lang_counts = Counter([(r.get("language") or {}).get("iso") for r in enriched]) or {"NA": len(enriched)}
    sent_counts = Counter([(r.get("sentiment") or {}).get("label") for r in enriched]) or {"NA": len(enriched)}
    LOG.info("[SUMMARY] extraidos=%d, escritos=%d", len(items), len(enriched))
    LOG.info("[SUMMARY] languages=%s", dict(lang_counts))
    LOG.info("[SUMMARY] sentiment=%s", dict(sent_counts))
    LOG.info("[SUMMARY] outputs: ndjson=%s preview_csv=%s heartbeat=%s",
             output_ndjson, output_preview, heartbeat)

# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import io
import csv
import sys
import json
import math
import yaml
import hashlib
import logging
import datetime as dt
from collections import defaultdict, Counter
from urllib.parse import urlparse

# deps opcionales
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    HAS_BLOB = True
except Exception:
    HAS_BLOB = False

# Cliente de Azure Language (tu archivo existente)
from news_etl.azure_language import AzureLanguageClient

# -----------------------
# Utils / configuración
# -----------------------

def load_config(path="config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def now_utc():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def to_date_utc(ts: dt.datetime, tz_name: str = "UTC") -> dt.date:
    # si quisieras usar pytz/zoneinfo para otra TZ, puedes ampliarlo.
    return ts.astimezone(dt.timezone.utc).date()

def parse_iso_datetime(s: str) -> dt.datetime:
    # Acepta ...Z
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        # fallback: intenta con strptime comunes
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"):
            try:
                return dt.datetime.strptime(s, fmt)
            except Exception:
                pass
        # si no, UTC now como último recurso
        return now_utc()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest()

def fix_mojibake(text: str) -> str:
    if not isinstance(text, str):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text

def host_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _host_in_suffix_list(host: str, blocked: list[str]) -> bool:
    host = (host or "").lower()
    for b in blocked or []:
        b = (b or "").lower().strip()
        if not b:
            continue
        if host == b or host.endswith("." + b):
            return True
    return False

# -----------------------
# Fetch (placeholder)
# -----------------------
#
# Esta función es un marcador. Usa tu propio fetch (RSS, APIs, etc.)
# Debe devolver una lista de items con:
#   { "published_at": iso8601, "source": str, "title": str, "summary": str|None, "url": str }
#
def fetch_news_items(cfg: dict) -> list[dict]:
    # En tu repositorio original probablemente ya tienes esta fase.
    # Aquí, asumimos que ya obtuviste una lista `items` antes de enriquecer.
    # Si ya la tienes, sustituye este bloque por tu llamada real.
    logging.warning("[FETCH] Usando fetch de ejemplo. Reemplaza con tu implementación real.")
    return []

# -----------------------
# Filtros previos
# -----------------------

def is_language_allowed(lang_iso: str, cfg: dict) -> bool:
    allow = (cfg.get("filters", {}) or {}).get("languages_allowlist") or []
    if not allow:
        return True
    return (lang_iso or "").lower() in [x.lower() for x in allow]

def is_commerce_spam(text: str, cfg: dict) -> bool:
    c = (cfg.get("filters", {}) or {}).get("commerce", {}) or {}
    price_re = [re.compile(rx) for rx in c.get("price_regex") or []]
    pct_re = [re.compile(rx) for rx in c.get("percent_regex") or []]
    min_price = int(c.get("min_price_tokens", 0) or 0)
    min_pct = int(c.get("min_percent_tokens", 0) or 0)

    price_hits = sum(1 for rx in price_re if rx.search(text))
    pct_hits = sum(1 for rx in pct_re if rx.search(text))
    if price_hits >= min_price or pct_hits >= min_pct:
        return True
    return False

def passes_source_domain_filters(item: dict, cfg: dict) -> bool:
    url = item.get("url", "")
    host = host_from_url(url)
    reg = cfg.get("registry", {}) or {}
    allow = reg.get("domains_allowlist") or []
    block = reg.get("domains_blacklist") or []

    # Allowlist (si hay)
    if allow:
        if not _host_in_suffix_list(host, allow):
            return False

    # Blacklist por sufijo
    if _host_in_suffix_list(host, block):
        return False

    # Blacklist por "source" textual
    src = (item.get("source") or "").lower().strip()
    sb = [s.lower() for s in (cfg.get("filters", {}) or {}).get("source_blacklist", [])]
    if src and src in sb:
        return False

    return True

# -----------------------
# Azure enrichment
# -----------------------

def build_azure_docs(items: list[dict]) -> list[dict]:
    docs = []
    for i, it in enumerate(items, start=1):
        title = fix_mojibake(it.get("title") or "")
        summary = fix_mojibake(it.get("summary") or "")
        text = f"{title}. {summary}".strip()
        docs.append({"id": str(i), "text": text})
    return docs

def enrich_with_azure(items: list[dict], cfg: dict) -> list[dict]:
    alcfg = cfg.get("azure_language", {}) or {}
    client = AzureLanguageClient(
        endpoint=alcfg.get("endpoint", ""),
        key=alcfg.get("key", os.getenv("AZURE_LANGUAGE_KEY", "")),
        api_version=alcfg.get("api_version", "2023-04-01"),
        force_language=alcfg.get("force_language"),
        timeout_seconds=int(alcfg.get("timeout_seconds", 30)),
    )
    flags = alcfg.get("tasks") or {}

    docs = build_azure_docs(items)
    grouped = client.analyze_batch(docs, tasks=flags)
    parsed = client.parse_results(grouped)  # id -> dict con language/sentiment/kp/entities/linked

    # acoplar resultados por índice
    out = []
    for i, it in enumerate(items, start=1):
        rid = str(i)
        res = parsed.get(rid, {})
        it2 = dict(it)  # copia
        it2["language"] = res.get("language")
        it2["sentiment"] = res.get("sentiment")
        it2["key_phrases"] = res.get("key_phrases", [])
        it2["entities"] = res.get("entities", [])
        it2["linked_entities"] = res.get("linked_entities", [])
        out.append(it2)
    return out

# -----------------------
# Ticker tagging
# -----------------------

def tag_tickers(item: dict, cfg: dict) -> list[str]:
    tcfg = cfg.get("ticker_tagging", {}) or {}
    if not tcfg.get("enabled", True):
        return []

    patterns = tcfg.get("tickers", {}) or {}
    negative = [re.compile(rx, re.I) for rx in (tcfg.get("negative_patterns") or [])]

    title = fix_mojibake(item.get("title") or "")
    summary = fix_mojibake(item.get("summary") or "")
    text = " ".join([title, summary, item.get("source") or ""])
    url = item.get("url", "")

    # cortar por patrones negativos
    if any(rx.search(text) or rx.search(url) for rx in negative):
        return []

    # entidades
    ents = [e.lower() for e in (item.get("entities") or [])]
    linked = [e.lower() for e in (item.get("linked_entities") or [])]

    found = set()

    # helper Android→Google (solo si también hay Google)
    def ok_android_for_google():
        t = (title + " " + summary).lower()
        if "android" not in t:
            return True
        return ("google" in t) or ("alphabet" in t) or any(le in ("google", "alphabet", "android") for le in linked)

    for ticker, regs in patterns.items():
        compiled = [re.compile(rx) for rx in regs or []]

        # chequeo texto/plano
        matched_text = any(rx.search(title) or rx.search(summary) for rx in compiled)

        # chequeo por entidades/linking
        joined_set = " ".join(ents + linked)
        matched_ent = any(rx.search(joined_set) for rx in compiled)

        if ticker == "GOOGL" and not ok_android_for_google():
            continue

        if matched_text or matched_ent:
            found.add(ticker)

    return sorted(found)

# -----------------------
# Ventana de 15 días + límite por día
# -----------------------

def in_last_n_days(published_at: str, n_days: int) -> bool:
    if not published_at:
        return False
    ts = parse_iso_datetime(published_at)
    now = now_utc()
    delta = now - ts
    return 0 <= delta.days <= max(0, n_days - 1) or (now.date() == ts.date())

def truncate_to_day(dt_iso: str, tz_name: str = "UTC") -> str:
    ts = parse_iso_datetime(dt_iso)
    d = to_date_utc(ts, tz_name)
    return d.isoformat()  # YYYY-MM-DD

def apply_date_and_limits(items: list[dict], cfg: dict) -> list[dict]:
    dw = cfg.get("date_window", {}) or {}
    days_back = int(dw.get("days_back", 15))
    tz_name = dw.get("tz", "UTC")

    limits = cfg.get("daily_limits", {}) or {}
    per_ticker_limit = int(limits.get("per_ticker_limit", 3))
    include_no_ticker = bool(limits.get("include_no_ticker", True))
    no_ticker_limit = int(limits.get("no_ticker_per_day_limit", 999))

    # 1) Filtrar por ventana móvil
    items15 = [it for it in items if in_last_n_days(it.get("published_at"), days_back)]

    # 2) Agrupar por día
    by_day: dict[str, list[dict]] = defaultdict(list)
    for it in items15:
        day = truncate_to_day(it.get("published_at"), tz_name)
        by_day[day].append(it)

    selected_all: list[dict] = []

    # 3) Por día, aplicar límites por ticker (sin perder los sin ticker)
    for day, bucket in by_day.items():
        # separar por ticker y sin ticker
        per_ticker: dict[str, list[dict]] = defaultdict(list)
        no_ticker: list[dict] = []

        for it in bucket:
            tks = it.get("tickers") or []
            if tks:
                for tk in set(tks):
                    per_ticker[tk].append(it)
            else:
                no_ticker.append(it)

        # Para evitar duplicados si un mismo item tiene varios tickers, vamos a ir marcando IDs
        emitted_ids = set()

        # 3a) limitar por ticker
        for tk, arr in per_ticker.items():
            # mantener orden estable por published_at descendente
            arr_sorted = sorted(arr, key=lambda x: x.get("published_at", ""), reverse=True)
            for it in arr_sorted[:per_ticker_limit]:
                _id = it.get("_row_id") or sha1(it.get("url", "") + (it.get("title") or ""))
                if _id in emitted_ids:
                    continue
                it["_selected_day"] = day
                selected_all.append(it)
                emitted_ids.add(_id)

        # 3b) incluir sin ticker (con su propio límite)
        if include_no_ticker and no_ticker_limit > 0:
            no_ticker_sorted = sorted(no_ticker, key=lambda x: x.get("published_at", ""), reverse=True)
            take = no_ticker_sorted[:no_ticker_limit]
            for it in take:
                _id = it.get("_row_id") or sha1(it.get("url", "") + (it.get("title") or ""))
                if _id in emitted_ids:
                    continue
                it["_selected_day"] = day
                selected_all.append(it)
                emitted_ids.add(_id)

    # Orden final por fecha descendente
    selected_all.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return selected_all

# -----------------------
# Pipeline principal
# -----------------------

def build_preview_csv(rows: list[dict], out_csv: str, max_rows: int = 1000):
    header = [
        "published_at","source","title","lang",
        "sentiment_label","sentiment_score","tickers","url",
        "key_phrases","entities","linked_entities"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows[:max_rows]:
            lang_iso = (r.get("language") or {}).get("iso")
            sent = r.get("sentiment") or {}
            tks = r.get("tickers") or []
            kps = r.get("key_phrases") or []
            ents = r.get("entities") or []
            links = r.get("linked_entities") or []

            w.writerow([
                r.get("published_at",""),
                r.get("source",""),
                fix_mojibake(r.get("title","")),
                lang_iso or "",
                sent.get("label") or "",
                f"{sent.get('score') or 0:.2f}",
                ",".join(tks),
                r.get("url",""),
                "|".join(kps),
                "|".join(ents),
                "|".join(links),
            ])

def upload_blob(cfg: dict, local_path: str, remote_name: str, content_type: str = "application/octet-stream"):
    ab = cfg.get("azure_blob", {}) or {}
    if not ab.get("enabled", False):
        return
    if not HAS_BLOB:
        logging.warning("[BLOB] azure-storage-blob no instalado; skip upload.")
        return

    account_url = ab.get("account_url")
    container = ab.get("container")
    prefix = ab.get("prefix_path", "").strip("/")

    sas = ab.get("sas_token") or os.getenv("AZ_BLOB_SAS_TOKEN", "")
    if sas and not sas.startswith("?"):
        sas = "?" + sas

    bsc = BlobServiceClient(account_url=account_url + sas)
    client = bsc.get_container_client(container)

    blob_name = f"{prefix}/{remote_name}" if prefix else remote_name
    logging.info("[BLOB] Subiendo %s -> %s", local_path, blob_name)
    with open(local_path, "rb") as f:
        client.upload_blob(
            name=blob_name,
            data=f,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )

def main():
    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, (cfg.get("logging", {}) or {}).get("level", "INFO")),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log = logging.getLogger(__name__)

    out_cfg = cfg.get("output", {}) or {}
    out_dir = out_cfg.get("dir", "data")
    ensure_dir(out_dir)

    # 1) Fetch
    raw_items = fetch_news_items(cfg)  # reemplaza por tu propio loader si ya lo tienes
    log.info("[FETCH] items crudos: %d", len(raw_items))

    # 2) Pre-filtrado por dominio/source y heurísticas básicas
    pre = []
    for it in raw_items:
        url = it.get("url","")
        title = it.get("title") or ""
        summary = it.get("summary") or ""
        if not url or not title:
            continue
        if not passes_source_domain_filters(it, cfg):
            continue
        text_for_commerce = " ".join([title, summary, url])
        if is_commerce_spam(text_for_commerce, cfg):
            continue
        pre.append(it)
    log.info("[FILTER] tras filtros iniciales: %d", len(pre))

    if not pre:
        log.warning("No hay items tras filtros. Saliendo.")
        return

    # 3) Enriquecimiento Azure
    enriched = enrich_with_azure(pre, cfg)
    log.info("[AZURE] enriquecidos: %d", len(enriched))

    # 4) Filtrar por idioma permitido (si está configurado)
    enriched2 = []
    for it in enriched:
        lang_iso = (it.get("language") or {}).get("iso")
        if is_language_allowed(lang_iso or "", cfg):
            enriched2.append(it)
    log.info("[LANG] tras language_allowlist: %d", len(enriched2))

    # 5) Ticker tagging
    for it in enriched2:
        it["tickers"] = tag_tickers(it, cfg)

    # 6) Aplicar ventana de 15 días y límites por día/ticker
    windowed = apply_date_and_limits(enriched2, cfg)
    log.info("[WINDOW] seleccionados tras 15 días + límites diarios: %d", len(windowed))

    # 7) Salidas (NDJSON + preview CSV)
    prefix = out_cfg.get("prefix", "news")
    ts_day = now_utc().date().isoformat().replace("-", "")
    ndjson_path = os.path.join(out_dir, f"{prefix}_{ts_day}.ndjson")
    with open(ndjson_path, "w", encoding="utf-8") as f:
        for it in windowed:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    if out_cfg.get("preview_csv", True):
        csv_path = os.path.join(out_dir, f"{prefix}_preview_{ts_day}.csv")
        build_preview_csv(windowed, csv_path, max_rows=int(out_cfg.get("preview_max_rows", 1000)))
    else:
        csv_path = None

    # 8) Heartbeat
    hb_path = os.path.join(out_dir, "heartbeat.txt")
    with open(hb_path, "w", encoding="utf-8") as f:
        f.write(now_utc().isoformat())

    # 9) Uploads
    if cfg.get("azure_blob", {}).get("enabled", False):
        # NDJSON
        upload_blob(
            cfg, ndjson_path,
            remote_name=os.path.basename(ndjson_path),
            content_type="application/x-ndjson",
        )
        if csv_path:
            upload_blob(
                cfg, csv_path,
                remote_name=os.path.basename(csv_path),
                content_type="text/csv",
            )
        upload_blob(
            cfg, hb_path,
            remote_name="heartbeat.txt",
            content_type="text/plain",
        )

    # 10) Summary
    langs = Counter([(it.get("language") or {}).get("iso") for it in windowed])
    sents = Counter([(it.get("sentiment") or {}).get("label") for it in windowed])
    log.info("[SUMMARY] seleccionados=%d", len(windowed))
    log.info("[SUMMARY] languages=%s", dict(langs))
    log.info("[SUMMARY] sentiment=%s", dict(sents))
    outs = f"ndjson={ndjson_path} preview_csv={csv_path or '-'} heartbeat={hb_path}"
    log.info("[SUMMARY] outputs: %s", outs)

if __name__ == "__main__":
    main()

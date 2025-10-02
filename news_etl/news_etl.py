# news_etl/news_etl.py
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

# Clientes/utilidades locales
from azure_language import AzureLanguageClient


# =========================
# Helpers de log y fechas
# =========================
LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def today_yyyymmdd(tz: Optional[str] = None) -> str:
    # usa hora local del runner
    return dt.datetime.now().strftime("%Y%m%d")


# ==================================================
# Expansión de variables de entorno dentro del YAML
# Soporta ${VAR} y ${VAR:-default} y también $VAR
# ==================================================
_ENV_PAT = re.compile(
    r"""
    ^\$\{?                  # ${ o $
    (?P<name>[A-Z0-9_]+)    # nombre
    (?:\:-(?P<default>.*))? # opcional ':-default'
    \}?$                    # opcierre
    """,
    re.VERBOSE,
)


def _expand_env_str(s: str) -> str:
    m = _ENV_PAT.match(s)
    if m:
        name = m.group("name")
        default = m.group("default")
        val = os.getenv(name, default if default is not None else "")
        # si no hay default y tampoco env definida, conserva el original
        return val if (val != "" or default is not None) else s
    # también permite strings mezclados con $VAR
    return os.path.expandvars(s)


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_str(obj)
    return obj


# =========================
# Estructuras / IO helpers
# =========================
@dataclass
class NewsItem:
    id: str
    published_at: str
    source: str
    title: str
    url: str


def load_input_ndjson(path: str, sample_limit: int = 0) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                LOGGER.warning("Línea %s inválida en NDJSON", i)
            if sample_limit and len(items) >= sample_limit:
                break
    LOGGER.info("Entradas: %d | tras dedupe: %d", len(items), len(items))
    return items


def write_ndjson(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_preview_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    # columnas más útiles para revisar
    cols = [
        "published_at",
        "source",
        "title",
        "lang",
        "sentiment_label",
        "sentiment_score",
        "key_phrases",
        "entities",
        "linked_entities",
        "tickers",
        "url",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =========================
# Filtros
# =========================
def normalize_host(url: str) -> str:
    try:
        from urllib.parse import urlparse

        netloc = urlparse(url).netloc.lower()
        # quita www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def apply_filters(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    filters = cfg.get("filters", {})
    commerce = cfg.get("commerce_filter", {})
    registry_black = set(map(str.lower, cfg.get("registry_domains_blacklist", [])))

    source_black = set(map(str.lower, filters.get("source_blacklist", [])))
    source_white = set(map(str.lower, filters.get("allowed_sources_whitelist", [])))
    exclude_patterns = [re.compile(p) for p in filters.get("exclude_phrases", [])]

    title_terms = [re.compile(p) for p in commerce.get("title_url_terms", [])]
    path_terms = [re.compile(p) for p in commerce.get("path_terms", [])]
    allow_corp = set(map(str.lower, commerce.get("allowed_corporate_domains", [])))

    def is_commerce_like(title: str, url: str) -> bool:
        uhost = normalize_host(url)
        upath = url
        # permitir blogs corporativos explícitamente
        if any(uhost.endswith(dom) for dom in allow_corp):
            return False
        if any(p.search(title) for p in title_terms):
            return True
        if any(p.search(url) for p in title_terms):
            return True
        if any(p.search(upath) for p in path_terms):
            return True
        return False

    out: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    for it in items:
        url = it.get("url", "")
        title = it.get("title", "")
        source = it.get("source", "")
        host = normalize_host(url)

        # dedupe por URL
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Whitelist tiene prioridad
        if source_white and source.lower() not in source_white:
            continue

        # Blacklists de fuente y registros de paquetes
        if source.lower() in source_black:
            continue
        if host in registry_black:
            continue

        # frases a excluir
        if any(p.search(title) for p in exclude_patterns):
            continue

        # filtro de comercio
        if commerce.get("enabled", False) and is_commerce_like(title, url):
            continue

        out.append(it)

    LOGGER.info("Tras filtros: %d", len(out))
    return out


# =====================================================
# Enriquecimiento con Azure Language (cliente por lote)
# =====================================================
def enrich_with_azure(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    lang_cfg = cfg.get("azure_language", {})
    endpoint = lang_cfg.get("endpoint") or ""
    key = lang_cfg.get("key") or ""
    api_version = lang_cfg.get("api_version", "2023-04-01")
    force_language = lang_cfg.get("force_language") or None
    tasks = lang_cfg.get("tasks", {})
    batch_size = int(lang_cfg.get("batch_size", 5))  # el API suele permitir 5 por request
    timeout = int(lang_cfg.get("timeout_seconds", 30))

    client = AzureLanguageClient(
        endpoint=endpoint,
        key=key,
        api_version=api_version,
        timeout_seconds=timeout,
    )

    # Prepara documentos por lote
    def chunk(seq: List[Dict[str, Any]], n: int):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    results: List[Dict[str, Any]] = []

    for batch in chunk(items, batch_size):
        docs = [
            {
                "id": str(i),
                "text": f"{it.get('title','')}. {it.get('source','')}",
                **({"language": force_language} if force_language else {}),
            }
            for i, it in enumerate(batch, start=1)
        ]

        # Llama tareas habilitadas, acumulando resultados por id
        merged: Dict[str, Dict[str, Any]] = {
            str(i): {} for i in range(1, len(batch) + 1)
        }

        if tasks.get("language_detection", True):
            lang_res = client.analyze_batch("LanguageDetection", docs)
            for d in lang_res.get("documents", []):
                merged[d["id"]]["lang"] = d.get("detectedLanguage", {}).get("iso6391Name")

        if tasks.get("sentiment_analysis", True):
            sent_res = client.analyze_batch("SentimentAnalysis", docs)
            for d in sent_res.get("documents", []):
                merged[d["id"]]["sentiment_label"] = d.get("sentiment")
                merged[d["id"]]["sentiment_score"] = float(
                    d.get("confidenceScores", {}).get(d.get("sentiment"), 0.0)
                )

        if tasks.get("key_phrase_extraction", True):
            kpe_res = client.analyze_batch("KeyPhraseExtraction", docs)
            for d in kpe_res.get("documents", []):
                merged[d["id"]]["key_phrases"] = "; ".join(d.get("keyPhrases", []))

        if tasks.get("entity_recognition", True):
            ner_res = client.analyze_batch("EntityRecognition", docs)
            for d in ner_res.get("documents", []):
                ents = [e.get("text") for e in d.get("entities", [])]
                merged[d["id"]]["entities"] = "; ".join(ents)

        if tasks.get("entity_linking", True):
            nel_res = client.analyze_batch("EntityLinking", docs)
            for d in nel_res.get("documents", []):
                ents = [e.get("name") for e in d.get("entities", [])]
                merged[d["id"]]["linked_entities"] = "; ".join(ents)

        # fusionar con los items originales del batch
        for idx, it in enumerate(batch, start=1):
            enr = merged[str(idx)]
            results.append({**it, **enr})

    return results


# =========================
# Ticker tagging simple
# =========================
def tag_tickers(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    tt_cfg = cfg.get("ticker_tagging", {})
    if not tt_cfg.get("enabled", False):
        return

    rules: Dict[str, List[re.Pattern]] = {}
    for ticker, patterns in tt_cfg.get("tickers", {}).items():
        rules[ticker] = [re.compile(p) for p in patterns]

    for it in items:
        text = f"{it.get('title','')} {it.get('source','')}"
        hits: List[str] = []
        for tick, pats in rules.items():
            if any(p.search(text) for p in pats):
                hits.append(tick)
        it["tickers"] = ", ".join(sorted(set(hits)))


# =========================
# Función principal
# =========================
def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    # Expande ${VAR} y ${VAR:-default}
    cfg = _expand_env(cfg_raw)

    setup_logging(cfg.get("runtime", {}).get("log_level", "INFO"))

    # IO
    io_cfg = cfg.get("io", {})
    date_str = today_yyyymmdd()
    input_path = io_cfg.get("input_ndjson", "data/news_raw.ndjson")
    output_ndjson = io_cfg.get("output_ndjson", "data/news_{{date}}.ndjson").replace("{{date}}", date_str)
    output_preview_csv = io_cfg.get("output_preview_csv", "data/news_preview_{{date}}.csv").replace("{{date}}", date_str)
    heartbeat_file = io_cfg.get("heartbeat_file", "data/heartbeat.txt")

    # sample_limit seguro
    sample_limit_raw = cfg.get("runtime", {}).get("sample_limit", 0)
    try:
        sample_limit = int(sample_limit_raw)
    except Exception:
        LOGGER.warning("sample_limit inválido '%s', usando 0", sample_limit_raw)
        sample_limit = 0

    LOGGER.info("Cargando input: %s", input_path)
    raw_items = load_input_ndjson(input_path, sample_limit)

    # Filtrar
    items = apply_filters(raw_items, cfg)

    # Enriquecer
    try:
        enriched = enrich_with_azure(items, cfg)
    except Exception:
        LOGGER.exception("Fallo enriqueciendo con Azure Language. Se salvarán sin enriquecimiento.")
        enriched = items

    # Tickers
    tag_tickers(enriched, cfg)

    # Salvar
    write_ndjson(output_ndjson, enriched)
    write_preview_csv(output_preview_csv, enriched)

    # Heartbeat
    Path(heartbeat_file).parent.mkdir(parents=True, exist_ok=True)
    Path(heartbeat_file).write_text(dt.datetime.now().isoformat(), encoding="utf-8")

    # Resumen
    langs = {}
    sents = {}
    for r in enriched:
        langs[r.get("lang")] = langs.get(r.get("lang"), 0) + 1
        sents[r.get("sentiment_label")] = sents.get(r.get("sentiment_label"), 0) + 1

    LOGGER.info("[SUMMARY] extraidos=%d, escritos=%d", len(raw_items), len(enriched))
    LOGGER.info("[SUMMARY] languages=%s", langs)
    LOGGER.info("[SUMMARY] sentiment=%s", sents)
    LOGGER.info(
        "[SUMMARY] outputs: ndjson=%s preview_csv=%s heartbeat=%s",
        output_ndjson,
        output_preview_csv,
        heartbeat_file,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta a config YAML")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)

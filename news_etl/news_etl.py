# news_etl/news_etl.py
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import logging
import os
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

# Opcional: subir a Azure Blob si está configurado
try:
    from azure.storage.blob import BlobServiceClient
    _AZURE_BLOB_AVAILABLE = True
except Exception:
    _AZURE_BLOB_AVAILABLE = False

# Cliente de Azure Language (tu archivo existente)
try:
    from .azure_language import AzureLanguageClient
except Exception:
    from azure_language import AzureLanguageClient  # type: ignore

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# ------------------------------
# Utilidades
# ------------------------------

def parse_iso8601(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

def to_iso8601(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _expand_env(obj: Any) -> Any:
    """
    Expande variables de entorno en strings ($VAR o ${VAR})
    recursivamente en dicts/listas. Usa os.path.expandvars.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj

# ------------------------------
# Config
# ------------------------------

def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración en '{cfg_path}'. "
            f"CWD='{Path.cwd()}'. Pasa --config con una ruta válida."
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _expand_env(raw)
    return cfg

# ------------------------------
# Modelos
# ------------------------------

@dataclass
class RawItem:
    published_at: datetime
    source: str
    title: str
    summary: Optional[str]
    url: str
    raw_tickers: List[str] = field(default_factory=list)

@dataclass
class EnrichedItem:
    published_at: datetime
    source: str
    title: str
    summary: Optional[str]
    url: str

    language: Dict[str, Any] = field(default_factory=dict)
    sentiment: Dict[str, Any] = field(default_factory=dict)
    key_phrases: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    linked_entities: List[str] = field(default_factory=list)

    tickers: List[str] = field(default_factory=list)

# ------------------------------
# Carga de input (raw NDJSON)
# ------------------------------

def load_raw_news(ndjson_path: Path) -> List[RawItem]:
    if not ndjson_path.exists():
        raise FileNotFoundError(f"No existe el input NDJSON: {ndjson_path}")

    items: List[RawItem] = []
    with ndjson_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            published_at = obj.get("published_at") or obj.get("published") or obj.get("date")
            if not published_at:
                continue

            try:
                dt = parse_iso8601(published_at)
            except Exception:
                continue

            items.append(
                RawItem(
                    published_at=dt,
                    source=str(obj.get("source") or obj.get("site") or ""),
                    title=str(obj.get("title") or ""),
                    summary=(obj.get("summary") or obj.get("description")),
                    url=str(obj.get("url") or obj.get("link") or ""),
                    raw_tickers=list(obj.get("raw_tickers") or obj.get("tickers") or []),
                )
            )
    return items

# ------------------------------
# Enriquecimiento (Azure Language)
# ------------------------------

def azure_enrich(
    items: List[RawItem],
    az_cfg: Dict[str, Any],
    batch_size: int = 5
) -> List[EnrichedItem]:

    flags = (az_cfg.get("tasks")
             or {"language_detection": True, "sentiment_analysis": True,
                 "key_phrase_extraction": True, "entity_recognition": True,
                 "entity_linking": True})

    endpoint = (az_cfg.get("endpoint") or "").strip()
    key = (az_cfg.get("key") or "").strip()
    if not endpoint or endpoint.startswith("${") or endpoint.startswith("$"):
        raise ValueError(
            "Azure Language 'endpoint' no configurado correctamente. "
            "Asegúrate de definir AZURE_LANGUAGE_ENDPOINT en el entorno "
            "o poner un valor literal en config.yaml."
        )

    client = AzureLanguageClient(
        endpoint=endpoint,
        key=key,
        api_version=az_cfg.get("api_version", "2023-04-01"),
        force_language=az_cfg.get("force_language"),
        timeout_seconds=int(az_cfg.get("timeout_seconds", 30)),
    )

    # Preparamos documentos: usamos title + summary para mejor señal
    docs: List[Dict[str, Any]] = []
    for idx, it in enumerate(items, start=1):
        text = it.title
        if it.summary:
            text = f"{it.title}. {it.summary}"
        docs.append({"id": str(idx), "text": text})

    # Llamada a Azure
    grouped = client.analyze_batch(docs, tasks=flags)
    per_doc = AzureLanguageClient.parse_results(grouped)

    # Mapear de vuelta
    enriched: List[EnrichedItem] = []
    for idx, it in enumerate(items, start=1):
        r = per_doc.get(str(idx), {})
        enriched.append(
            EnrichedItem(
                published_at=it.published_at,
                source=it.source,
                title=it.title,
                summary=it.summary,
                url=it.url,
                language=r.get("language") or {},
                sentiment=r.get("sentiment") or {},
                key_phrases=r.get("key_phrases") or [],
                entities=r.get("entities") or [],
                linked_entities=r.get("linked_entities") or [],
                tickers=list(it.raw_tickers),
            )
        )
    return enriched

# ------------------------------
# Tickers (por entidades)
# ------------------------------

def build_ticker_matcher(cfg: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    mapping = cfg.get("ticker_matching", {})
    out: List[Tuple[str, List[str]]] = []
    for ticker, syns in mapping.items():
        syns_lc = [s.strip().lower() for s in (syns or []) if s and isinstance(s, str)]
        if syns_lc:
            out.append((ticker, syns_lc))
    return out

def infer_tickers(it: EnrichedItem, matcher: List[Tuple[str, List[str]]]) -> List[str]:
    haystack = set()
    for s in [it.title, it.summary or ""]:
        if s:
            # split básico + versiones completas para captar frases
            haystack.update(w.strip().lower() for w in s.split())
            haystack.add(s.strip().lower())

    for l in (it.entities or []):
        haystack.add(l.lower().strip())
    for l in (it.linked_entities or []):
        haystack.add(l.lower().strip())
    for l in (it.key_phrases or []):
        haystack.add(l.lower().strip())

    out = set(it.tickers or [])
    for ticker, syns in matcher:
        if any(any(k in token for token in haystack) for k in syns):
            out.add(ticker)
    return sorted(out)

# ------------------------------
# Filtrado por ventana temporal y límites diarios
# ------------------------------

def filter_by_date(items: List[EnrichedItem], days_back: int) -> List[EnrichedItem]:
    if days_back <= 0:
        return items
    today_utc = datetime.now(timezone.utc)
    start_dt = today_utc - timedelta(days=days_back)
    return [it for it in items if it.published_at >= start_dt]

def limit_per_day_and_ticker(
    items: List[EnrichedItem],
    per_ticker_limit: int,
    include_no_ticker: bool,
    no_ticker_per_day_limit: int,
) -> List[EnrichedItem]:
    """
    Máximo N por (día, ticker). Maneja 'sin ticker' aparte.
    Una nota con varios tickers cuenta para cada ticker; se mantiene una sola vez.
    """
    if per_ticker_limit <= 0 and (not include_no_ticker or no_ticker_per_day_limit <= 0):
        return []

    items_sorted = sorted(items, key=lambda x: x.published_at, reverse=True)

    taken_flags: Dict[int, bool] = {}
    count_by_day_ticker: Dict[Tuple[str, str], int] = Counter()
    count_no_ticker_by_day: Dict[str, int] = Counter()

    for idx, it in enumerate(items_sorted):
        day = it.published_at.astimezone(timezone.utc).date().isoformat()
        if it.tickers:
            added = False
            for tk in it.tickers:
                key = (day, tk)
                if count_by_day_ticker[key] < per_ticker_limit:
                    count_by_day_ticker[key] += 1
                    added = True
            if added:
                taken_flags[idx] = True
        else:
            if include_no_ticker and no_ticker_per_day_limit > 0:
                if count_no_ticker_by_day[day] < no_ticker_per_day_limit:
                    count_no_ticker_by_day[day] += 1
                    taken_flags[idx] = True

    kept = [items_sorted[i] for i in sorted(taken_flags.keys())]
    kept = sorted(kept, key=lambda x: x.published_at)
    return kept

# ------------------------------
# Salida (NDJSON + CSV)
# ------------------------------

def write_outputs(
    items: List[EnrichedItem],
    out_dir: Path,
    prefix: str = "news"
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    today_tag = datetime.now(timezone.utc).strftime("%Y%m%d")

    ndjson_path = out_dir / f"{prefix}_{today_tag}.ndjson"
    csv_path = out_dir / f"{prefix}_preview_{today_tag}.csv"

    with ndjson_path.open("w", encoding="utf-8") as f:
        for it in items:
            obj = {
                "published_at": to_iso8601(it.published_at),
                "source": it.source,
                "title": it.title,
                "summary": it.summary,
                "url": it.url,
                "language": it.language,
                "sentiment": it.sentiment,
                "key_phrases": it.key_phrases,
                "entities": it.entities,
                "linked_entities": it.linked_entities,
                "tickers": it.tickers,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    headers = [
        "published_at", "source", "title", "lang",
        "sentiment_label", "sentiment_score",
        "tickers", "url",
        "key_phrases", "entities", "linked_entities",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for it in items:
            lang_iso = (it.language or {}).get("iso")
            sent = it.sentiment or {}
            label = sent.get("label")
            score = sent.get("score")
            w.writerow({
                "published_at": to_iso8601(it.published_at),
                "source": it.source,
                "title": it.title,
                "lang": lang_iso,
                "sentiment_label": label,
                "sentiment_score": f"{score:.2f}" if isinstance(score, (int, float)) else "",
                "tickers": ",".join(it.tickers),
                "url": it.url,
                "key_phrases": "|".join(it.key_phrases),
                "entities": "|".join(it.entities),
                "linked_entities": "|".join(it.linked_entities),
            })

    LOG.info("[OUTPUT] NDJSON=%s CSV=%s", ndjson_path, csv_path)
    return ndjson_path, csv_path

# ------------------------------
# Azure Blob (opcional)
# ------------------------------

def maybe_upload_to_blob(
    cfg: Dict[str, Any],
    files: Dict[str, Path],
    remote_prefix: str = "news"
) -> None:
    storage = cfg.get("storage", {}) or {}
    if not storage.get("enabled", False):
        return
    if not _AZURE_BLOB_AVAILABLE:
        LOG.warning("azure-storage-blob no disponible; omitimos upload.")
        return

    conn_str = storage.get("connection_string")
    container = storage.get("container")
    if not conn_str or not container:
        LOG.warning("[BLOB] Falta connection_string o container; omitimos upload.")
        return

    bsc = BlobServiceClient.from_connection_string(conn_str)
    client = bsc.get_container_client(container)

    for logical_name, path in files.items():
        blob_name = f"{remote_prefix}/{path.name}"
        LOG.info("[BLOB] Subiendo %s -> %s", path, blob_name)
        with path.open("rb") as f:
            client.upload_blob(name=blob_name, data=f, overwrite=True)

    hb_path = Path("data/heartbeat.txt")
    hb_path.write_text(f"{datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    client.upload_blob(
        name=f"{remote_prefix}/heartbeat.txt",
        data=hb_path.open("rb"),
        overwrite=True
    )
    LOG.info("[BLOB] Heartbeat subido.")

# ------------------------------
# MAIN
# ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ETL de noticias (enriquecimiento + filtros).")
    parser.add_argument("--config", required=True, help="Ruta al archivo config.yaml")
    parser.add_argument("--print-config-path", action="store_true")
    parser.add_argument("--days-back", type=int, default=None, help="Últimos N días (override)")
    parser.add_argument("--per-ticker-limit", type=int, default=None, help="Máximo por ticker y por día (override)")
    parser.add_argument("--include-no-ticker", action="store_true", help="Incluir notas sin ticker (override ON)")
    parser.add_argument("--exclude-no-ticker", action="store_true", help="Excluir notas sin ticker (override OFF)")
    parser.add_argument("--no-ticker-per-day-limit", type=int, default=None, help="Máximo por día para sin ticker")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.print_config_path:
        LOG.info("[ETL] Usando configuración desde: %s", Path(args.config).resolve())

    inputs_cfg = cfg.get("inputs", {}) or {}
    raw_path = Path(inputs_cfg.get("raw_ndjson", "data/news_raw.ndjson"))

    enrich_cfg = cfg.get("azure_language", {}) or {}
    days_back = args.days_back if args.days_back is not None else int(cfg.get("days_back", 1))

    limits_cfg = cfg.get("daily_limits", {}) or {}
    per_ticker_limit = args.per_ticker_limit if args.per_ticker_limit is not None else int(limits_cfg.get("per_ticker_limit", 3))
    include_no_ticker = limits_cfg.get("include_no_ticker", True)
    if args.include_no_ticker:
        include_no_ticker = True
    if args.exclude_no_ticker:
        include_no_ticker = False
    no_ticker_per_day_limit = args.no_ticker_per_day_limit if args.no_ticker_per_day_limit is not None else int(limits_cfg.get("no_ticker_per_day_limit", 1))

    out_dir = Path(cfg.get("outputs", {}).get("dir", "data"))
    out_prefix = cfg.get("outputs", {}).get("prefix", "news")

    # 1) Cargar RAW
    LOG.info("[STEP] Leyendo RAW desde %s", raw_path)
    raw_items = load_raw_news(raw_path)
    LOG.info("[SUMMARY] RAW cargado: %d items", len(raw_items))

    # 2) Enriquecer (Azure)
    LOG.info("[STEP] Enriqueciendo con Azure Language…")
    enriched = azure_enrich(raw_items, enrich_cfg)
    LOG.info("[SUMMARY] Enriquecidos: %d", len(enriched))

    # 3) Tickers (matcher por config)
    matcher = build_ticker_matcher(cfg)
    for it in enriched:
        it.tickers = infer_tickers(it, matcher)

    # 4) Filtro temporal
    before_filter = len(enriched)
    enriched = filter_by_date(enriched, days_back=days_back)
    LOG.info("[FILTER] Últimos %d días: %d -> %d", days_back, before_filter, len(enriched))

    # 5) Límites diarios por ticker (+ sin ticker opcional)
    before_limit = len(enriched)
    limited = limit_per_day_and_ticker(
        enriched,
        per_ticker_limit=per_ticker_limit,
        include_no_ticker=include_no_ticker,
        no_ticker_per_day_limit=no_ticker_per_day_limit,
    )
    LOG.info(
        "[LIMIT] per_ticker=%d include_no_ticker=%s no_ticker/day=%d: %d -> %d",
        per_ticker_limit, include_no_ticker, no_ticker_per_day_limit, before_limit, len(limited)
    )

    # 6) Salida
    ndjson_path, csv_path = write_outputs(limited, out_dir=out_dir, prefix=out_prefix)

    # 7) Métricas simples
    langs = Counter((it.language or {}).get("iso", "??") for it in limited)
    sentiments = Counter((it.sentiment or {}).get("label", "unknown") for it in limited)
    LOG.info("[SUMMARY] escritos=%d", len(limited))
    LOG.info("[SUMMARY] languages=%s", dict(langs))
    LOG.info("[SUMMARY] sentiment=%s", dict(sentiments))

    # 8) Upload opcional
    try:
        maybe_upload_to_blob(cfg, {"ndjson": ndjson_path, "preview_csv": csv_path}, remote_prefix=cfg.get("outputs", {}).get("blob_prefix", "news"))
    except Exception as ex:
        LOG.warning("Upload a blob falló (no crítico): %s", ex)

if __name__ == "__main__":
    main()

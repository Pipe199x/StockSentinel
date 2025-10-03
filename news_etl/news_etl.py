# news_etl/news_etl.py
import argparse
import json
import os
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import glob
import csv
import logging

from .azure_language import AzureLanguageClient

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m):
            var = m.group(1)
            return os.getenv(var, "")
        return _ENV_VAR_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso_ts(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc)

def to_iso_date(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%d")

def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg or {})

def _read_one_ndjson(file_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def read_raw_items(path_like: str) -> List[Dict[str, Any]]:
    """
    Acepta:
      - archivo NDJSON
      - patrón glob (p.ej. data/raw/news_*.ndjson)
      - directorio (lee *.ndjson dentro)
    """
    p = Path(path_like)
    files: List[Path] = []
    if "*" in path_like or "?" in path_like or "[" in path_like:
        files = [Path(x) for x in glob.glob(path_like)]
    elif p.is_dir():
        files = sorted(p.glob("*.ndjson"))
    else:
        files = [p] if p.exists() else []

    if not files:
        LOG.warning("[RAW] No se encontraron archivos con '%s'", path_like)
        return []

    out: List[Dict[str, Any]] = []
    for fp in sorted(files):
        out.extend(_read_one_ndjson(fp))
    LOG.info("[SUMMARY] RAW cargado: %s items desde %s archivo(s)", len(out), len(files))
    return out

def write_ndjson(path: str, items: Iterable[Dict[str, Any]]) -> Path:
    p = Path(path)
    _ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return p

def write_preview_csv(path: str, rows: List[Dict[str, Any]]) -> Path:
    p = Path(path)
    _ensure_parent(p)
    cols = [
        "published_at", "source", "title", "lang",
        "sentiment_label", "sentiment_score", "tickers",
        "url", "key_phrases", "entities", "linked_entities"
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p

def azure_enrich(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not cfg.get("enabled", False) or not items:
        return items

    endpoint = cfg.get("endpoint", "").rstrip("/")
    key = cfg.get("key", "")
    api_version = cfg.get("api_version", "2023-04-01")
    force_language = cfg.get("force_language") or None
    timeout_seconds = int(cfg.get("timeout_seconds", 30))
    flags = cfg.get("tasks", {})

    if not endpoint or not key:
        LOG.warning("[AZURE] endpoint/key faltantes; se omite enriquecimiento.")
        return items

    client = AzureLanguageClient(
        endpoint=endpoint,
        key=key,
        api_version=api_version,
        force_language=force_language,
        timeout_seconds=timeout_seconds,
    )

    docs = []
    for i, it in enumerate(items, start=1):
        text = " ".join(
            str(x) for x in [
                it.get("title") or "",
                it.get("summary") or "",
                it.get("source") or "",
            ] if x
        )
        docs.append({"id": str(i), "text": text})

    grouped = client.analyze_batch(docs, tasks=flags)
    per_doc = AzureLanguageClient.parse_results(grouped)

    out = []
    for i, it in enumerate(items, start=1):
        dres = per_doc.get(str(i), {})
        lang = dres.get("language") or {}
        sent = dres.get("sentiment") or {}
        kps = dres.get("key_phrases") or []
        ents = dres.get("entities") or []
        links = dres.get("linked_entities") or []
        enriched = dict(it)
        if lang:
            enriched["language"] = {
                "iso": lang.get("iso"),
                "name": lang.get("name"),
                "score": lang.get("score"),
            }
        if sent:
            enriched["sentiment"] = {
                "label": sent.get("label"),
                "score": sent.get("score", 0.0),
            }
        enriched["key_phrases"] = kps
        enriched["entities"] = ents
        enriched["linked_entities"] = links
        out.append(enriched)
    return out

@dataclass
class TickerUniverse:
    restrict: bool
    mapping: Dict[str, List[str]]

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "TickerUniverse":
        uni = cfg.get("universe", {}) or {}
        norm: Dict[str, List[str]] = {}
        for tkr, aliases in uni.items():
            aliases = aliases or []
            norm[tkr] = [a.lower() for a in aliases] + [tkr.lower()]
        return cls(
            restrict=bool(cfg.get("restrict_to_universe", True)),
            mapping=norm,
        )

    def tag(self, text_tokens: List[str]) -> List[str]:
        text_set = {t.lower() for t in text_tokens if t}
        found: List[str] = []
        for tkr, aliases in self.mapping.items():
            if any(a in text_set for a in aliases):
                found.append(tkr)
        return sorted(set(found))

def detect_tickers(item: Dict[str, Any], uni: TickerUniverse) -> List[str]:
    tokens: List[str] = []
    tokens += [item.get("title", ""), item.get("summary", "")]
    tokens += item.get("entities", [])
    tokens += item.get("linked_entities", [])
    tokens = [t for t in tokens if isinstance(t, str)]
    lowered = []
    for t in tokens:
        for piece in re.split(r"[^\w\-\.\&]+", t):
            piece = piece.strip().lower()
            if piece:
                lowered.append(piece)
    tickers = uni.tag(lowered)
    return tickers

def filter_last_n_days(items: List[Dict[str, Any]], today_utc: datetime, n_days: int) -> List[Dict[str, Any]]:
    if n_days <= 0:
        return items
    start = (today_utc - timedelta(days=n_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = today_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
    out: List[Dict[str, Any]] = []
    for it in items:
        ts = it.get("published_at") or it.get("publishedAt") or it.get("date")
        if not ts:
            continue
        try:
            dt = parse_iso_ts(ts)
        except Exception:
            continue
        if start <= dt <= end:
            out.append(it)
    return out

def _safe_dt(it: Dict[str, Any]) -> datetime:
    ts = it.get("published_at") or ""
    try:
        return parse_iso_ts(ts)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

def select_daily_limited(items: List[Dict[str, Any]],
                         per_ticker_limit: int,
                         include_no_ticker: bool,
                         no_ticker_per_day_limit: int) -> List[Dict[str, Any]]:
    """
    Limita a N por ticker por día. Ordena por fecha desc dentro de cada bucket.
    Además, realiza una verificación final estricta para no exceder cupos.
    """
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for it in items:
        ts = it.get("published_at")
        if not ts:
            continue
        dt = _safe_dt(it)
        dkey = to_iso_date(dt)
        tickers = it.get("tickers") or []
        if tickers:
            for t in sorted(set(tickers)):
                by_key[(dkey, t)].append(it)
        else:
            by_key[(dkey, "__NONE__")].append(it)

    prelim: List[Dict[str, Any]] = []
    seen_dedup = set()
    for (dkey, tkr), bucket in sorted(by_key.items()):
        bucket.sort(key=_safe_dt, reverse=True)
        limit = (0 if (tkr == "__NONE__" and not include_no_ticker)
                 else (no_ticker_per_day_limit if tkr == "__NONE__" else per_ticker_limit))
        limit = max(0, int(limit))
        if limit == 0:
            continue
        for it in bucket[:limit*2]:
            key = (it.get("url") or "") + "|" + (it.get("title") or "")
            if key in seen_dedup:
                continue
            seen_dedup.add(key)
            prelim.append(it)

    caps: Dict[Tuple[str, str], int] = defaultdict(int)
    final: List[Dict[str, Any]] = []
    for it in sorted(prelim, key=_safe_dt, reverse=True):
        dt = _safe_dt(it)
        dkey = to_iso_date(dt)
        tks = it.get("tickers") or []
        if not tks:
            if not include_no_ticker:
                continue
            if caps[(dkey, "__NONE__")] >= no_ticker_per_day_limit:
                continue
            caps[(dkey, "__NONE__")] += 1
            final.append(it)
            continue

        would_exceed = any(caps[(dkey, t)] >= per_ticker_limit for t in set(tks))
        if would_exceed:
            continue
        for t in set(tks):
            caps[(dkey, t)] += 1
        final.append(it)

    return final

def build_preview_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for it in items:
        lang_iso = (it.get("language") or {}).get("iso") or ""
        sent = it.get("sentiment") or {}
        ticks = it.get("tickers") or []
        rows.append({
            "published_at": it.get("published_at") or "",
            "source": it.get("source") or "",
            "title": it.get("title") or "",
            "lang": lang_iso,
            "sentiment_label": sent.get("label") or "",
            "sentiment_score": sent.get("score") if isinstance(sent.get("score"), (int, float)) else "",
            "tickers": ",".join(ticks),
            "url": it.get("url") or "",
            "key_phrases": "|".join(it.get("key_phrases") or []),
            "entities": "|".join(it.get("entities") or []),
            "linked_entities": "|".join(it.get("linked_entities") or []),
        })
    return rows

def upload_to_blob(cfg: Dict[str, Any], local_path: Path, dest_name: str) -> None:
    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except Exception:
        LOG.warning("[BLOB] azure-storage-blob no instalado; omitiendo upload.")
        return
    conn = cfg.get("connection_string") or ""
    if not conn:
        LOG.warning("[BLOB] Connection string vacía; omitiendo upload.")
        return
    container = cfg.get("container", "datasets")
    prefix = cfg.get("prefix", "")
    bsc = BlobServiceClient.from_connection_string(conn)
    client = bsc.get_container_client(container)
    client.create_container(exist_ok=True)
    blob_name = f"{prefix}{dest_name}"
    with local_path.open("rb") as data:
        client.upload_blob(name=blob_name, data=data, overwrite=True)
    LOG.info("[BLOB] Subido %s -> %s/%s", local_path, container, blob_name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta a config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    tz = timezone.utc
    if cfg.get("runtime", {}).get("today_override"):
        today = parse_iso_ts(cfg["runtime"]["today_override"])
    else:
        today = datetime.now(tz)

    p_raw = cfg["paths"]["raw_ndjson"]
    p_out_ndjson_tpl = cfg["paths"]["out_ndjson"]
    p_out_csv_tpl = cfg["paths"]["out_preview_csv"]
    p_heartbeat = cfg["paths"]["heartbeat"]

    LOG.info("[STEP] Leyendo RAW desde %s", p_raw)
    raw_all = read_raw_items(p_raw)

    LOG.info("[STEP] Enriqueciendo con Azure Language…")
    enriched = azure_enrich(raw_all, cfg.get("azure_language", {}))

    uni = TickerUniverse.from_cfg(cfg.get("tickers", {}))
    for it in enriched:
        it["tickers"] = detect_tickers(it, uni)

    n_days = int(cfg.get("window", {}).get("last_n_days", 15))
    filtered = filter_last_n_days(enriched, today, n_days)

    dl = cfg.get("daily_limits", {})
    per_ticker = int(dl.get("per_ticker_limit", 3))
    include_no_ticker = bool(dl.get("include_no_ticker", False))
    no_ticker_limit = int(dl.get("no_ticker_per_day_limit", 0))
    limited = select_daily_limited(filtered, per_ticker, include_no_ticker, no_ticker_limit)

    limited.sort(key=lambda x: x.get("published_at", ""), reverse=True)

    stamp = today.strftime("%Y%m%d")
    out_ndjson = p_out_ndjson_tpl.replace("YYYYMMDD", stamp)
    out_csv = p_out_csv_tpl.replace("YYYYMMDD", stamp)

    LOG.info("[STEP] Escribiendo NDJSON y CSV…")
    ndjson_path = write_ndjson(out_ndjson, limited)
    preview_rows = build_preview_rows(limited)
    max_rows = int(cfg.get("limits", {}).get("preview_csv_max_rows", 1000))
    csv_path = write_preview_csv(out_csv, preview_rows[:max_rows])

    hb_path = Path(p_heartbeat)
    _ensure_parent(hb_path)
    hb_path.write_text(iso_utc_now() + "\n", encoding="utf-8")
    LOG.info("[OUTPUT] heartbeat=%s", hb_path)

    LOG.info(
        "[SUMMARY] extraidos=%d, tras_ventana=%d, seleccionados=%d",
        len(raw_all), len(filtered), len(limited)
    )
    langs = Counter([(it.get("language") or {}).get("iso") for it in limited if it.get("language")])
    sents = Counter([(it.get("sentiment") or {}).get("label") for it in limited if it.get("sentiment")])
    LOG.info("[SUMMARY] languages=%s", dict(langs))
    LOG.info("[SUMMARY] sentiment=%s", dict(sents))
    LOG.info("[SUMMARY] outputs: ndjson=%s preview_csv=%s heartbeat=%s", ndjson_path, csv_path, hb_path)

    bcfg = cfg.get("blob_upload", {}) or {}
    if bcfg.get("enabled"):
        LOG.info("[BLOB] Subiendo archivos (modo ETL)…")
        upload_to_blob(bcfg, ndjson_path, f"news_{stamp}.ndjson")
        upload_to_blob(bcfg, csv_path, f"news_preview_{stamp}.csv")
        upload_to_blob(bcfg, hb_path, "heartbeat.txt")

if __name__ == "__main__":
    main()

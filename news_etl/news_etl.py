# news_etl/news_etl.py — REPLACE COMPLETE
# Publica histórico (news/news_YYYYMMDD.ndjson, preview) y alias estable (news/latest/*)

import argparse
import json
import os
import re
import glob
import csv
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .finbert_sentiment import finbert_enrich
from .news_client import _dedupe_by_url
from etl_common.manifest import update_manifest

LOG = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# =============================
# Utilidades de config/tiempo
# =============================
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

def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg or {})

def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso_ts(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc)

def to_iso_date(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%d")

# =============================
# I/O archivos locales
# =============================

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    if any(ch in path_like for ch in "*?["):
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

# =============================
# Tagging de tickers
# =============================
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

# =============================
# Selección por ventana y límites diarios
# =============================

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
    Dedup por (url,title).
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
    seen = set()
    for (dkey, tkr), bucket in sorted(by_key.items()):
        bucket.sort(key=_safe_dt, reverse=True)
        limit = (0 if (tkr == "__NONE__" and not include_no_ticker)
                 else (no_ticker_per_day_limit if tkr == "__NONE__" else per_ticker_limit))
        limit = max(0, int(limit))
        if limit == 0:
            continue
        for it in bucket:
            key = (it.get("source") or "") + "|" + (it.get("url") or it.get("title") or "")
            if key in seen:
                continue
            seen.add(key)
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

# =============================
# Salida preview CSV
# =============================

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

# =============================
# Publicación a docs/ (GitHub Pages)
# =============================

def write_latest_json(path: str, items: List[Dict[str, Any]], today: datetime, window_days: int) -> Path:
    p = Path(path)
    _ensure_parent(p)
    payload = {
        "as_of": iso_utc_now(),
        "window_days": window_days,
        "articles": items,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p

# =============================
# MAIN
# =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta a config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Fecha de referencia (UTC)
    if cfg.get("runtime", {}).get("today_override"):
        today = parse_iso_ts(cfg["runtime"]["today_override"]).astimezone(timezone.utc)
    else:
        today = datetime.now(timezone.utc)

    # Paths
    p_raw = cfg["paths"]["raw_ndjson"]
    p_out_ndjson_tpl = cfg["paths"]["out_ndjson"]
    p_out_csv_tpl = cfg["paths"]["out_preview_csv"]
    p_heartbeat = cfg["paths"]["heartbeat"]
    p_history = cfg["paths"].get("history_ndjson", "docs/data/history/news_history.ndjson")
    p_latest_json = cfg["paths"].get("latest_json", "docs/data/news_latest.json")
    p_manifest = cfg["paths"].get("manifest", "docs/data/manifest.json")

    # 1) Lee RAW (fetch incremental del día)
    LOG.info("[STEP] Leyendo RAW desde %s", p_raw)
    raw_all = read_raw_items(p_raw)

    # 2) Lee histórico acumulado (ya enriquecido en corridas previas)
    history_path = Path(p_history)
    history_items = _read_one_ndjson(history_path) if history_path.exists() else []
    LOG.info("[STEP] Histórico: %d items desde %s", len(history_items), p_history)

    # 3) Tagging tickers sobre lo nuevo
    uni = TickerUniverse.from_cfg(cfg.get("tickers", {}))
    for it in raw_all:
        it["tickers"] = detect_tickers(it, uni)

    # 4) Merge + dedupe por URL (histórico primero: conserva versiones ya enriquecidas)
    combined = _dedupe_by_url(history_items + raw_all)

    # 5) Ventana de N días
    n_days = int(cfg.get("window", {}).get("last_n_days", 15))
    filtered = filter_last_n_days(combined, today, n_days)

    # 6) Límite por día/ticker
    dl = cfg.get("daily_limits", {})
    per_ticker = int(dl.get("per_ticker_limit", 3))
    include_no_ticker = bool(dl.get("include_no_ticker", False))
    no_ticker_limit = int(dl.get("no_ticker_per_day_limit", 0))
    limited = select_daily_limited(filtered, per_ticker, include_no_ticker, no_ticker_limit)

    # 7) Sentimiento FinBERT — solo sobre la selección final que aún no tiene score
    limited = finbert_enrich(limited, cfg.get("sentiment", {}))

    # Orden final por fecha desc
    limited.sort(key=lambda x: x.get("published_at", ""), reverse=True)

    # 8) Salidas locales con fecha
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

    # 9) Publicación a docs/ (GitHub Pages)
    latest_path = write_latest_json(p_latest_json, limited, today, n_days)
    new_history = _dedupe_by_url(limited + history_items)
    new_history.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    history_out = write_ndjson(p_history, new_history)
    update_manifest(p_manifest, "news", {
        "last_updated_utc": iso_utc_now(),
        "articles": len(limited),
        "history_articles": len(new_history),
        "window_days": n_days,
    })
    LOG.info("[OUTPUT] latest_json=%s history=%s (%d items)", latest_path, history_out, len(new_history))

    # 10) Logs de resumen
    LOG.info("[SUMMARY] raw=%d, historico=%d, tras_ventana=%d, seleccionados=%d",
             len(raw_all), len(history_items), len(filtered), len(limited))
    sents = Counter([(it.get("sentiment") or {}).get("label") for it in limited if it.get("sentiment")])
    LOG.info("[SUMMARY] sentiment=%s", dict(sents))
    LOG.info("[SUMMARY] outputs: ndjson=%s preview_csv=%s heartbeat=%s", ndjson_path, csv_path, hb_path)


if __name__ == "__main__":
    main()

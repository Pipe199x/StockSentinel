"""Shared manifest.json helper.

Each pipeline (news, stocks) owns one section of docs/data/manifest.json and
updates only that section, so the two daily runs never clobber each other's
metadata.
"""
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_MANIFEST: Dict[str, Any] = {
    "project": "StockSentinel",
    "tickers": ["AMZN", "GOOGL", "MSFT"],
    "news": None,
    "stocks": None,
}


def update_manifest(path: str, section: str, payload: Dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            manifest = dict(DEFAULT_MANIFEST)
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = dict(DEFAULT_MANIFEST)
    manifest[section] = payload
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p

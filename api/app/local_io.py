"""Local file backend: reads the same docs/data/ folder GitHub Pages serves.

DATA_DIR defaults to <repo root>/docs/data, so `uvicorn api.app.main:app`
run from the repo root works with no configuration.
"""
import json
import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "docs" / "data"))


def read_json(name: str):
    p = DATA_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_text(name: str) -> str | None:
    p = DATA_DIR / name
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")

"""Fail if any published JSON file is not strictly valid JSON.

`json.load` alone accepts NaN/Infinity; browsers do not. Run before committing
docs/data so a bad payload never reaches GitHub Pages.
"""
import json
import sys
from pathlib import Path


def _reject(const):
    raise ValueError(f"non-finite constant {const}")


def main(root="docs/data"):
    bad = 0
    for p in sorted(Path(root).rglob("*.json")):
        try:
            json.loads(p.read_text(encoding="utf-8"), parse_constant=_reject)
            print(f"[OK]   {p}")
        except ValueError as e:
            bad += 1
            print(f"[FAIL] {p}: {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))

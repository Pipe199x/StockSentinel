"""The live configuration's hash is pinned: only an intentional change under `quant:` may move it.

Every distinct hash is a trial the Deflated Sharpe pays for. Reporting knobs therefore live
outside `quant:` (strategy_label, report:, gates:, stress:). Update LIVE_HASH only in the
commit that deliberately switches the live strategy.
"""
import yaml

from stocks_etl.validation import config_hash

LIVE_HASH = "c3f2b4fc9bfd"   # V4 + news features (M3), live since 2026-11-12


def _live_quant():
    with open("stocks_etl/config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _hash(q):
    return config_hash({k: q[k] for k in sorted(q) if k != "mc_paths"})   # as meta_model.make_signals does


def test_live_hash_is_pinned():
    assert _hash(_live_quant()["quant"]) == LIVE_HASH


def test_sections_outside_quant_do_not_move_the_hash():
    cfg = _live_quant()
    cfg.update(strategy_label="anything", report={"x": 1}, gates={"y": 2}, stress={"z": 3})
    assert _hash(cfg["quant"]) == LIVE_HASH


def test_any_quant_change_is_a_new_trial():
    q = dict(_live_quant()["quant"])
    assert _hash(dict(q, k_tp=2.0)) != LIVE_HASH
    assert _hash(dict(q, mc_paths=1)) == LIVE_HASH          # Monte Carlo resolution is not a strategy change

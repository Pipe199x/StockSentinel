"""Paywall: the public JSON must not reveal today's picks (ticker, levels, odds, size).

The full files go to docs/data/premium/, which only the Cloudflare Pages Function serves to
license holders. A leak here would put the paid content in a public, cacheable file.
"""
import json

from stocks_etl.publish_json import redact_paper, redact_signals, write_public_and_premium

SECRET_SIGNAL_KEYS = {"ticker", "sector", "last_close", "tp", "sl", "p_success", "kelly_size", "thesis",
                      "sigma_garch", "var_95", "es_99", "mc_p50", "zscore", "hurst"}


def _signals():
    base = {"as_of": "2026-11-13", "market_regime": "normal", "market_p_stress": 0.1, "strategy": "momentum",
            "t_max_days": 20, "t_max_date": "2026-12-11", "regime": "normal", "sigma_garch": 0.01}
    picks = [{**base, "ticker": t, "sector": "X", "side": 1, "kelly_size": k, "tp": 101.5, "sl": 95.2,
              "p_success": 0.57, "last_close": 98.0, "thesis": f"{t} trend"} for t, k in [("NEE", 0.036), ("JPM", 0.034)]]
    idle = [{**base, "ticker": t, "sector": "Y", "side": 0, "kelly_size": 0.0, "tp": None, "sl": None,
             "p_success": 0.4, "last_close": 50.0, "strategy": "none"} for t in ("AAPL", "KO", "XOM")]
    return picks + idle


def test_public_signals_name_no_asset():
    out = redact_signals(_signals())
    text = json.dumps(out)
    for t in ("NEE", "JPM", "AAPL", "KO", "XOM"):  # idle names too: elimination would reveal the picks
        assert t not in text
    assert out["n_assets"] == 5 and out["n_idle"] == 3 and len(out["picks"]) == 2
    for p in out["picks"]:
        assert not SECRET_SIGNAL_KEYS & set(p)
        assert p["side"] == 1 and p["t_max_date"] == "2026-12-11"


def test_public_paper_hides_open_positions():
    paper = {"total_return": -0.001, "equity_curve": [{"date": "2026-11-12", "equity": 0.999}],
             "positions": [{"ticker": "NEE", "side": 1, "size": 0.036, "entry_price": 86.9, "tp": 94.8, "sl": 75.8,
                            "status": "open", "exit_by": "2026-12-10"}],
             "rescore": [{"ticker": "NEE", "p_profit_now": 0.38}], "recent_closed": []}
    out = redact_paper(paper)
    text = json.dumps(out)
    assert "NEE" not in text and "rescore" not in out
    assert out["positions"] == [{"side": 1, "status": "open", "exit_by": "2026-12-10"}]
    assert abs(out["invested"] - 0.036) < 1e-12 and out["total_return"] == -0.001


def test_writer_splits_full_and_public(tmp_path):
    write_public_and_premium(_signals(), str(tmp_path), "signals_latest.json", redact_signals)
    full = json.loads((tmp_path / "premium" / "signals_latest.json").read_text(encoding="utf-8"))
    public = json.loads((tmp_path / "signals_latest.json").read_text(encoding="utf-8"))
    assert {r["ticker"] for r in full} == {"NEE", "JPM", "AAPL", "KO", "XOM"}
    assert "NEE" not in json.dumps(public)

    # A premium-only file removes any stale public copy.
    (tmp_path / "risk_report.json").write_text("{}", encoding="utf-8")
    write_public_and_premium({"x": 1}, str(tmp_path), "risk_report.json")
    assert not (tmp_path / "risk_report.json").exists()
    assert (tmp_path / "premium" / "risk_report.json").exists()

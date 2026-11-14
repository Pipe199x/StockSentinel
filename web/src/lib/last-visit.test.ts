// Run: node --experimental-strip-types --test src/lib/last-visit.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { diffVisit, takeSnapshot } from "./last-visit.ts";

const sig = (ticker: string, kelly_size = 0.05, side = 1, as_of = "2026-11-13") => ({ ticker, as_of, side, kelly_size });
const row = (ticker: string, exit_reason: string, ret_net: number) => ({ ticker, signal_date: "2026-11-11", exit_reason, ret_net });

test("first visit shows nothing", () => {
  const d = diffVisit(null, [sig("AAPL")], [row("HD", "tp", 0.02)]);
  assert.equal(d.firstVisit, true);
  assert.deepEqual(d.newSignals, []);
  assert.equal(d.closed.length, 0);
});

test("only unseen active signals and newly closed trades count", () => {
  const prev = takeSnapshot([sig("AAPL"), sig("XOM", 0)], [row("HD", "tp", 0.02)]);
  const d = diffVisit(
    prev,
    [sig("AAPL"), sig("XOM"), sig("KO", 0), sig("AAPL", 0.05, -1, "2026-11-14")],
    [row("HD", "tp", 0.02), row("JPM", "tp", 0.03), row("NEE", "sl", -0.01)],
  );
  assert.deepEqual(d.newSignals, ["XOM", "AAPL"]); // XOM went active; AAPL flipped side on a new date
  assert.deepEqual(d.closed.map((r) => r.ticker), ["JPM", "NEE"]);
  assert.equal(d.wins, 1);
  assert.equal(d.losses, 1);
  assert.equal(d.tpHits, 1);
});

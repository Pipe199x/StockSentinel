/**
 * Custom: "since your last visit" — a per-browser snapshot of what the visitor already saw,
 * diffed against today's data. Pure functions (no React, no `@/` value imports) so the check
 * in last-visit.test.ts runs under plain `node --test`.
 */

export type SignalLike = { ticker: string; as_of: string; side: number; kelly_size: number };
export type ClosedLike = { ticker: string; signal_date: string; exit_reason: string | null; ret_net: number | null };

export type Snapshot = { seenAt: string; signals: string[]; closed: string[] };

export type VisitDiff = {
  firstVisit: boolean;
  seenAt: string | null;
  newSignals: string[]; // tickers
  closed: ClosedLike[];
  wins: number;
  losses: number;
  tpHits: number;
};

const sigKey = (s: SignalLike) => `${s.ticker}:${s.as_of}:${s.side}`;
const closedKey = (r: ClosedLike) => `${r.ticker}:${r.signal_date}`;

export function takeSnapshot(signals: SignalLike[], closed: ClosedLike[], now = new Date()): Snapshot {
  return {
    seenAt: now.toISOString(),
    signals: signals.filter((s) => s.kelly_size > 0).map(sigKey),
    closed: closed.map(closedKey),
  };
}

export function diffVisit(prev: Snapshot | null, signals: SignalLike[], closed: ClosedLike[]): VisitDiff {
  if (!prev) return { firstVisit: true, seenAt: null, newSignals: [], closed: [], wins: 0, losses: 0, tpHits: 0 };
  const seenSig = new Set(prev.signals);
  const seenClosed = new Set(prev.closed);
  const newClosed = closed.filter((r) => !seenClosed.has(closedKey(r)));
  return {
    firstVisit: false,
    seenAt: prev.seenAt,
    newSignals: signals.filter((s) => s.kelly_size > 0 && !seenSig.has(sigKey(s))).map((s) => s.ticker),
    closed: newClosed,
    wins: newClosed.filter((r) => (r.ret_net ?? 0) > 0).length,
    losses: newClosed.filter((r) => (r.ret_net ?? 0) < 0).length,
    tpHits: newClosed.filter((r) => r.exit_reason === "tp").length,
  };
}

const KEY = "ss:last-visit:v1";

/** Reads the previous snapshot; null on first visit, private mode or blocked storage. */
export function readSnapshot(): Snapshot | null {
  try {
    const raw = localStorage.getItem(KEY);
    const s = raw ? (JSON.parse(raw) as Snapshot) : null;
    return s && Array.isArray(s.signals) && Array.isArray(s.closed) ? s : null;
  } catch {
    return null;
  }
}

export function writeSnapshot(s: Snapshot) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {}
}

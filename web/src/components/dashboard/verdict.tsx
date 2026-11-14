"use client";

/** Hero content for the landing page: today's call in one line, plus fact chips. */
import { motion } from "motion/react";
import { IconSparkles } from "@tabler/icons-react";
import { ago, fin, longDate, REGIME } from "@/lib/format";
import type { Signal, ValidationReport } from "@/lib/data";
import type { VisitDiff } from "@/lib/last-visit";
import { ACCOUNT, money } from "./story";

const AND = new Intl.ListFormat("en", { type: "conjunction" });

/** "Long HD, JPM, and NEE. Short TLT." */
function sideSentence(active: Signal[]) {
  const by = (side: number) => active.filter((s) => s.side === side).map((s) => s.ticker);
  return (
    [
      [by(1), "Long"],
      [by(-1), "Short"],
    ] as [string[], string][]
  )
    .filter(([t]) => t.length)
    .map(([t, w]) => `${w} ${AND.format(t)}.`)
    .join(" ");
}

export function heroCopy(signals: Signal[], locked = false) {
  const active = signals.filter((s) => s.kelly_size > 0).sort((a, b) => b.kelly_size - a.kelly_size);
  const stress = signals[0]?.market_regime === "stress";
  const n = stress ? 0 : active.length;
  return {
    active: stress ? [] : active,
    title: n ? `${n} pick${n > 1 ? "s" : ""} today` : "No picks today",
    lead: stress
      ? "The market is in a stress period, and StockSentinel does not trade in stress periods. Not advice."
      : n
        ? locked
          ? "Unlock to see them. Not advice."
          : `${sideSentence(active)} Not advice.`
        : "No asset met StockSentinel's rules today. Not advice.",
  };
}

const chip = "inline-flex items-center gap-1.5 rounded-full bg-(--folio-card) px-3 py-1 text-xs text-(--folio-muted) ring-1 ring-black/8 dark:ring-white/10";

/** Market mood + data date, shown above the headline. */
export function HeroEyebrow({ signals, report }: { signals: Signal[]; report: ValidationReport | null }) {
  const mkt = signals[0]?.market_regime;
  return (
    <div className="flex flex-wrap justify-center gap-2">
      <span className={chip}>Market mood: {REGIME[mkt ?? ""]?.toLowerCase() ?? "—"}</span>
      {report && <span className={chip}>Data as of {longDate(report.as_of)}</span>}
    </div>
  );
}

/** Fact chips under the buttons: the account's since-start figure and what changed since the last visit. */
export function HeroFacts({ report, visit }: { report: ValidationReport | null; visit: VisitDiff | null }) {
  const cal = report?.meta.calendar;
  const from = report?.folds?.[0]?.test_from;
  const v = visit;
  const changes =
    v && !v.firstVisit
      ? [v.newSignals.length && `${v.newSignals.length} new pick${v.newSignals.length > 1 ? "s" : ""}`, v.wins && `${v.wins} win${v.wins > 1 ? "s" : ""}`, v.losses && `${v.losses} loss${v.losses > 1 ? "es" : ""}`].filter(Boolean)
      : [];
  return (
    <div className="flex flex-wrap justify-center gap-2">
      {cal && fin(cal.total_return) && (
        <span className={chip}>
          {money(ACCOUNT)} since {from ? longDate(from) : "the start"} → <span className="font-semibold text-(--folio-ink)">{money(ACCOUNT * (1 + cal.total_return))}</span> today
        </span>
      )}
      {v && !v.firstVisit && (
        <motion.span initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }} className={chip}>
          <IconSparkles className="h-3.5 w-3.5" />
          {changes.length ? `Since ${ago(v.seenAt)}: ${changes.join(", ")}` : `Nothing new since ${ago(v.seenAt)}`}
        </motion.span>
      )}
    </div>
  );
}

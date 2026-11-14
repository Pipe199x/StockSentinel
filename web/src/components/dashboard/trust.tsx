"use client";

/**
 * Technical details: Validation / Risk / Models behind three tabs, collapsed until asked.
 * Custom segmented control in the Aceternity Tabs style (shared-layout pill); the Aceternity
 * Tabs stack fixed-height cards, which doesn't fit these tall panels.
 */
import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import type { DeepReport, RiskReport, ValidationReport } from "@/lib/data";
import { Validation } from "./validation";
import { Risk } from "./risk";
import { Models } from "./models";
import { PanelBare } from "./panel";
import { LockedBlock } from "./paywall";

const TABS = [
  { key: "validation", label: "Validation" },
  { key: "risk", label: "Risk" },
  { key: "models", label: "Models" },
] as const;
type Key = (typeof TABS)[number]["key"];

/** Locked stand-in: a dummy grid of figures (no real values) behind the blur. */
function PlaceholderPanel() {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {Array.from({ length: 8 }, (_, i) => (
        <div key={i} className="rounded-xl p-4 ring-1 ring-black/8 dark:ring-white/10">
          <p className="text-xs text-neutral-500">Statistic name</p>
          <p className="mt-1 text-2xl font-semibold tabular-nums">0.000</p>
        </div>
      ))}
    </div>
  );
}

export function TechnicalDetails({ report, risk, deep, locked = false }: { report: ValidationReport | null; risk: RiskReport | null; deep: DeepReport | null; locked?: boolean }) {
  const [open, setOpen] = useState<Key | null>(null);
  return (
    <div>
      <div role="tablist" aria-label="Technical details" className="flex flex-wrap gap-1">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            role="tab"
            aria-selected={open === tab.key}
            onClick={() => setOpen(open === tab.key ? null : tab.key)}
            className="relative rounded-full px-4 py-2 text-sm font-medium ring-1 ring-black/8 focus-visible:ring-2 focus-visible:ring-cyan-500 focus-visible:outline-none dark:ring-white/10"
          >
            {open === tab.key && <motion.span layoutId="tech-pill" transition={{ type: "spring", bounce: 0.3, duration: 0.6 }} className="absolute inset-0 rounded-full bg-neutral-900/[0.06] dark:bg-white/[0.08]" />}
            <span className="relative">{tab.label}</span>
          </button>
        ))}
        {!open && <span className="self-center pl-2 text-xs text-neutral-500">Pick one to open the details.</span>}
      </div>

      <AnimatePresence mode="wait" initial={false}>
        {open && (
          <motion.div key={open} role="tabpanel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }} className="mt-6">
            {locked ? (
              <LockedBlock label="Unlock the technical details">
                <PlaceholderPanel />
              </LockedBlock>
            ) : (
              <PanelBare.Provider value>
                {open === "validation" && <Validation report={report} />}
                {open === "risk" && <Risk risk={risk} />}
                {open === "models" && <Models deep={deep} />}
              </PanelBare.Provider>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

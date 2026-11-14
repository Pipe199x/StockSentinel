"use client";

import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import { num, signedPct } from "@/lib/format";
import type { DeepModel, DeepReport } from "@/lib/data";
import { HoverEffect } from "@/components/ui/card-hover-effect";
import { Panel, Pill, Stat } from "./panel";

function ModelCard({ name, blurb, m }: { name: string; blurb: string; m?: DeepModel & { status?: string } }) {
  const promoted = !!m?.promoted;
  return (
    <BentoGridItem
      className="dark:bg-[#1c1c1a]"
      title={
        <span className="flex items-center gap-2">
          {name}
          <Pill tone={promoted ? "good" : m?.status === "insufficient_data" ? "warn" : "neutral"}>
            {promoted ? "Passes the gate (informational)" : m?.status === "insufficient_data" ? "Not enough data" : "Below the gate"}
          </Pill>
        </span>
      }
      description={blurb}
      header={
        m && fin3(m.dsr) ? (
          <HoverEffect className="grid grid-cols-2 gap-1 sm:grid-cols-3">
            <Stat label="DSR" value={num(m.dsr, 3)} tone={(m.dsr ?? 0) > 0.95 ? "good" : undefined} />
            <Stat label="EV / trade" value={signedPct(m.expected_value, 3)} />
            <Stat label="Trades" value={m.n_trades?.toLocaleString() ?? "—"} />
          </HoverEffect>
        ) : (
          <p className="text-sm text-neutral-500">No evaluation yet.</p>
        )
      }
    />
  );
}
const fin3 = (v: unknown) => typeof v === "number" && Number.isFinite(v);

export function Models({ deep }: { deep: DeepReport | null }) {
  if (!deep) return null;
  return (
    <Panel id="models" eyebrow="Weekly · gate DSR > 0.95 · informational, nothing trades on it" title="Deep models">
      <BentoGrid className="max-w-none md:auto-rows-auto">
        <ModelCard name="XGBoost (baseline)" blurb="The live meta-model, re-evaluated on the same out-of-sample events." m={deep.xgb} />
        <ModelCard name="LSTM" blurb="Sequence memory over 20 sessions of stationary features. It passes the gate only if it beats the live XGBoost on the same events and clears the DSR bar." m={deep.lstm} />
        <ModelCard name="Price + news fusion" blurb="FinBERT embeddings joined with price by cross-attention, not early fusion. Waits for enough news history." m={deep.fusion as DeepModel & { status?: string }} />
      </BentoGrid>
    </Panel>
  );
}

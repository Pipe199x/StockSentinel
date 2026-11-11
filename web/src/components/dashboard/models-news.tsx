"use client";

import { useState } from "react";
import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import { Tooltip } from "@/components/ui/tooltip-card";
import { HoverBorderGradient } from "@/components/ui/hover-border-gradient";
import { ago, num, pct, safeUrl, signedPct } from "@/lib/format";
import type { Article, DeepModel, DeepReport, News } from "@/lib/data";
import { HoverEffect } from "@/components/ui/card-hover-effect";
import { Panel, Pill, Stat } from "./panel";

function ModelCard({ name, blurb, m }: { name: string; blurb: string; m?: DeepModel & { status?: string } }) {
  const promoted = !!m?.promoted;
  return (
    <BentoGridItem
      className="dark:bg-[#101124]"
      title={
        <span className="flex items-center gap-2">
          {name}
          <Pill tone={promoted ? "good" : m?.status === "insufficient_data" ? "warn" : "neutral"}>
            {promoted ? "Promoted" : m?.status === "insufficient_data" ? "Not enough data" : "Not promoted"}
          </Pill>
        </span>
      }
      description={blurb}
      header={
        m && fin3(m.dsr) ? (
          <HoverEffect className="grid grid-cols-3 gap-1">
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
    <Panel id="models" eyebrow="Weekly · promotion gate DSR > 0.95" title="Deep models">
      <BentoGrid className="max-w-none md:auto-rows-auto">
        <ModelCard name="XGBoost (baseline)" blurb="The live meta-model, re-evaluated on the same out-of-sample events." m={deep.xgb} />
        <ModelCard name="LSTM" blurb="Sequence memory over 20 sessions of stationary features. Promoted only if it beats XGBoost and clears the DSR bar." m={deep.lstm} />
        <ModelCard name="Price + news fusion" blurb="FinBERT embeddings joined with price by cross-attention, not early fusion. Waits for enough news history." m={deep.fusion as DeepModel & { status?: string }} />
      </BentoGrid>
    </Panel>
  );
}

const SENT_TONE = { positive: "good", negative: "bad", neutral: "neutral" } as const;

function ArticleCard({ a }: { a: Article }) {
  const href = safeUrl(a.url);
  const s = a.sentiment;
  const body = (
    <article className="group/news h-full rounded-xl border border-neutral-200 bg-white/60 p-3 transition hover:border-cyan-500/40 dark:border-white/[0.08] dark:bg-[#101124]/60">
      <div className="flex items-center justify-between gap-2 text-xs text-neutral-500">
        <span className="truncate">{a.source} · {ago(a.published_at)}</span>
        {s && <Pill tone={SENT_TONE[s.label]}>{s.label} {pct(s.score, 0)}</Pill>}
      </div>
      <h3 className="mt-1.5 text-sm leading-snug font-medium group-hover/news:text-cyan-700 dark:group-hover/news:text-cyan-300">{a.title}</h3>
      {a.tickers?.length > 0 && <p className="mt-1.5 text-xs text-neutral-500">{a.tickers.join(" · ")}</p>}
    </article>
  );
  const linked = href ? <a href={href} target="_blank" rel="noopener noreferrer" className="block focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 rounded-xl">{body}</a> : body;
  return a.summary ? <Tooltip content={a.summary} containerClassName="block w-full">{linked}</Tooltip> : linked;
}

export function NewsFeed({ news }: { news: News | null }) {
  const [shown, setShown] = useState(8);
  if (!news) return null;
  const list = [...news.articles].sort((a, b) => b.published_at.localeCompare(a.published_at));
  const scored = list.filter((a) => a.sentiment);
  const pos = scored.filter((a) => a.sentiment!.label === "positive").length;
  const neg = scored.filter((a) => a.sentiment!.label === "negative").length;
  const mood = scored.length ? (pos - neg) / scored.length : 0;
  return (
    <Panel
      id="news"
      eyebrow="FinBERT sentiment"
      title="Top news"
      aside={<Pill tone={mood > 0.1 ? "good" : mood < -0.1 ? "bad" : "neutral"}>Mood {mood >= 0 ? "+" : ""}{num(mood, 2)}</Pill>}
    >
      <HoverEffect className="gap-1 md:grid-cols-2 lg:grid-cols-2">
        {list.slice(0, shown).map((a) => (
          <ArticleCard key={a.url + a.published_at} a={a} />
        ))}
      </HoverEffect>
      {shown < list.length && (
        <div className="mt-4 flex justify-center">
          <HoverBorderGradient
            as="button"
            onClick={() => setShown((n) => n + 8)}
            containerClassName="rounded-full"
            className="flex items-center gap-2 bg-white text-sm text-neutral-800 dark:bg-[#0c0d1c] dark:text-white"
          >
            Show more news
          </HoverBorderGradient>
        </div>
      )}
    </Panel>
  );
}

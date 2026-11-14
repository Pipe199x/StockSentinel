"use client";

import { cn } from "@/lib/utils";
import { ago, safeUrl } from "@/lib/format";
import type { Article, News } from "@/lib/data";
import { InfiniteMovingCards } from "@/components/ui/infinite-moving-cards";

const DOT = { positive: "bg-emerald-500", negative: "bg-rose-500", neutral: "bg-neutral-400" } as const;
const SENT = { positive: "Positive", negative: "Negative", neutral: "Neutral" } as const;
const PILL = {
  positive: "bg-emerald-500/12 text-emerald-700 dark:text-emerald-300",
  negative: "bg-rose-500/12 text-rose-700 dark:text-rose-300",
  neutral: "bg-neutral-500/12 text-neutral-600 dark:text-neutral-300",
} as const;

export const newest = (news: News | null) => [...(news?.articles ?? [])].sort((a, b) => b.published_at.localeCompare(a.published_at));

/** One headline: sentiment dot, tickers, title. A click opens the article; no hover popups. */
export function Headline({ a, compact }: { a: Article; compact?: boolean }) {
  const href = safeUrl(a.url);
  const s = a.sentiment?.label;
  const Tag = href ? "a" : "div";
  if (compact) {
    // Testimonial-card style (Aceternity SaaS template) for the headline loop.
    return (
      <Tag
        {...(href ? { href, target: "_blank", rel: "noopener noreferrer" } : {})}
        className="group/h flex h-full w-[19rem] flex-col justify-between gap-5 rounded-2xl bg-white p-5 ring-1 ring-black/8 transition hover:ring-black/20 focus-visible:ring-2 focus-visible:ring-cyan-500 focus-visible:outline-none sm:w-[24rem] dark:bg-[#1c1c1a] dark:ring-white/10 dark:hover:ring-white/25"
      >
        <span className="line-clamp-3 text-base leading-snug text-pretty text-neutral-800 dark:text-neutral-200">{a.title}</span>
        <span className="flex items-center gap-2.5 text-xs text-neutral-500">
          {s && (
            <span className={cn("inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-0.5 font-medium", PILL[s])}>
              <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", DOT[s])} />
              {SENT[s]}
            </span>
          )}
          {a.tickers?.length ? <span className="font-semibold text-neutral-900 dark:text-white">{a.tickers.slice(0, 3).join(" · ")}</span> : null}
          <span className="truncate">{a.source} · {ago(a.published_at)}</span>
        </span>
      </Tag>
    );
  }
  return (
    <Tag
      {...(href ? { href, target: "_blank", rel: "noopener noreferrer" } : {})}
      className={cn(
        "group/h flex h-full w-full items-start gap-2.5 rounded-xl border border-neutral-200 bg-white/70 p-3 transition-colors hover:border-cyan-500/50 focus-visible:ring-2 focus-visible:ring-cyan-500 focus-visible:outline-none dark:border-white/[0.08] dark:bg-[#1c1c1a]/70",
      )}
    >
      <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", s ? DOT[s] : "bg-neutral-300")} aria-label={s ? `${SENT[s]} sentiment` : undefined} role={s ? "img" : undefined} />
      <span className="min-w-0">
        <span className="block text-[11px] text-neutral-500">
          {a.tickers?.length ? <span className="font-semibold text-neutral-700 dark:text-neutral-300">{a.tickers.slice(0, 3).join(" · ")}</span> : null}
          {a.tickers?.length ? " · " : ""}
          {ago(a.published_at)}
        </span>
        <span className="mt-0.5 block text-sm leading-snug font-medium group-hover/h:text-cyan-700 dark:group-hover/h:text-cyan-300">{a.title}</span>
      </span>
    </Tag>
  );
}

/** Slow marquee of the newest headlines, near the top of the page. */
export function HeadlineStrip({ news }: { news: News | null }) {
  const list = newest(news).slice(0, 20);
  if (!list.length) return null;
  return (
    <div className="relative left-1/2 w-screen -translate-x-1/2">
      <InfiniteMovingCards key={news?.as_of} items={list} itemKey={(a) => a.url + a.published_at} renderItem={(a) => <Headline a={a} compact />} speed="slow" label="headlines" itemClassName="flex" />
    </div>
  );
}

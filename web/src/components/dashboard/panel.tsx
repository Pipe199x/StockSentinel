"use client";

import { createContext, useContext } from "react";
import { cn } from "@/lib/utils";
import { GlowingEffect } from "@/components/ui/glowing-effect";

/** Inside a container that already draws the frame (the Trust tabs), panels drop their own. */
export const PanelBare = createContext(false);

/**
 * Section shell: Aceternity GlowingEffect border on the same surface as the BentoGridItem
 * cards. The wrapper itself is only layout (title row + content).
 */
export function Panel({
  id,
  eyebrow,
  title,
  aside,
  children,
  className,
}: {
  id?: string;
  eyebrow?: string;
  title: string;
  aside?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  const bare = useContext(PanelBare);
  return (
    <section
      id={id}
      aria-labelledby={id ? `${id}-h` : undefined}
      className={cn(
        "relative scroll-mt-28",
        !bare && "rounded-2xl border border-neutral-200 bg-white p-5 backdrop-blur-sm md:p-6 dark:border-white/[0.1] dark:bg-[#1c1c1a]",
        className,
      )}
    >
      {!bare && <GlowingEffect spread={40} glow disabled={false} proximity={64} inactiveZone={0.01} borderWidth={2} />}
      <header className="relative mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          {eyebrow && <p className="text-xs font-semibold tracking-[0.18em] text-cyan-700 uppercase dark:text-cyan-300/80">{eyebrow}</p>}
          <h2 id={id ? `${id}-h` : undefined} className="mt-1 text-xl font-semibold tracking-tight md:text-2xl">
            {title}
          </h2>
        </div>
        {aside}
      </header>
      <div className="relative">{children}</div>
    </section>
  );
}

export function Pill({ children, tone = "neutral", className }: { children: React.ReactNode; tone?: "neutral" | "good" | "bad" | "warn" | "info"; className?: string }) {
  const t = {
    neutral: "border-neutral-200 text-neutral-600 dark:border-white/10 dark:text-neutral-300",
    good: "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
    bad: "border-rose-500/30 bg-rose-500/10 text-rose-700 dark:text-rose-300",
    warn: "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300",
    info: "border-cyan-500/30 bg-cyan-500/10 text-cyan-800 dark:text-cyan-200",
  }[tone];
  return <span className={cn("inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium whitespace-nowrap", t, className)}>{children}</span>;
}

export function Stat({ label, value, sub, tone }: { label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: "good" | "bad" }) {
  return (
    <div className="rounded-xl border border-neutral-200 p-3 dark:border-white/[0.08]">
      <p className="text-[11px] font-medium tracking-wide text-neutral-500 uppercase">{label}</p>
      <p className={cn("mt-1 text-lg font-semibold tabular-nums", tone === "good" && "text-emerald-600 dark:text-emerald-400", tone === "bad" && "text-rose-600 dark:text-rose-400")}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-neutral-500">{sub}</p>}
    </div>
  );
}

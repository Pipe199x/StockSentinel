"use client";
import { cn } from "@/lib/utils";
import React, { useRef } from "react";
import { motion, useInView, useSpring, useTransform } from "motion/react";
import { useEffect } from "react";

export type StatItem = {
  label: string;
  value: number | null;
  /** Formats the animated value (e.g. percent, 3 decimals). */
  format?: (v: number) => string;
  /** Plain text shown instead of a number (e.g. "Calm"). */
  text?: string;
  description: string;
  tone?: "good" | "bad" | "neutral";
};

/** Aceternity "Stats With Number Ticker", adapted to take live KPIs. */
export default function StatsWithNumberTicker({ items, className }: { items: StatItem[]; className?: string }) {
  return (
    <section className={cn("group/container relative w-full", className)}>
      <div className="relative z-20 grid grid-cols-2 gap-x-4 gap-y-5 sm:gap-6 lg:grid-cols-4">
        {items.map((item, index) => (
          <motion.div
            initial={{ y: 20, opacity: 0, filter: "blur(4px)" }}
            animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            key={item.label}
            className={cn("group/card relative overflow-hidden rounded-lg")}
          >
            <p className="text-[10px] font-semibold tracking-[0.14em] text-cyan-700 uppercase sm:text-xs sm:tracking-[0.18em] dark:text-cyan-300/80">
              {item.label}
            </p>
            <div className="mt-2 flex items-center gap-2">
              <p
                className={cn(
                  "text-2xl font-bold text-neutral-700 tabular-nums sm:text-3xl dark:text-neutral-200",
                  item.tone === "good" && "text-emerald-600 dark:text-emerald-400",
                  item.tone === "bad" && "text-rose-600 dark:text-rose-400",
                )}
              >
                {item.text ?? (item.value == null ? "—" : <AnimatedNumber value={item.value} format={item.format} />)}
              </p>
            </div>
            <p className="mt-2 text-xs text-balance text-neutral-600 sm:mt-3 sm:text-sm dark:text-neutral-300">{item.description}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

function AnimatedNumber({
  value,
  initial = 0,
  format,
}: {
  value: number;
  initial?: number;
  format?: (v: number) => string;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref);

  const spring = useSpring(initial, { mass: 0.8, stiffness: 75, damping: 15 });
  const display = useTransform(spring, (current) =>
    format ? format(current) : Math.round(current).toLocaleString(),
  );

  useEffect(() => {
    if (isInView) {
      spring.set(value);
    } else {
      spring.set(initial);
    }
  }, [isInView, spring, value, initial]);

  return <motion.span ref={ref}>{display}</motion.span>;
}

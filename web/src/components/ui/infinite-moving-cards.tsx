"use client";

/**
 * Aceternity "Infinite Moving Cards", adapted:
 * - renders any item (not just quotes) and clones once, even under StrictMode;
 * - always browsable by hand: arrow buttons, swipe, trackpad. Using an arrow stops the
 *   auto-scroll so the visitor stays in control;
 * - loops even under prefers-reduced-motion (owner's choice): it is slow, pauses on hover,
 *   and the arrows stop it, which covers WCAG 2.2.2 (pause, stop, hide).
 */
import { cn } from "@/lib/utils";
import React, { useEffect, useRef, useState } from "react";
import { IconChevronLeft, IconChevronRight } from "@tabler/icons-react";

export function InfiniteMovingCards<T>({
  items,
  renderItem,
  itemKey,
  direction = "left",
  speed = "slow",
  pauseOnHover = true,
  className,
  itemClassName,
  label = "items",
}: {
  items: T[];
  renderItem: (item: T) => React.ReactNode;
  itemKey: (item: T, idx: number) => string;
  direction?: "left" | "right";
  speed?: "fast" | "normal" | "slow";
  pauseOnHover?: boolean;
  className?: string;
  itemClassName?: string;
  label?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollerRef = useRef<HTMLUListElement>(null);
  const cloned = useRef(false);
  const [start, setStart] = useState(false);
  const [manual, setManual] = useState(false);

  useEffect(() => {
    const c = containerRef.current;
    const s = scrollerRef.current;
    if (!c || !s || cloned.current) return;
    cloned.current = true;
    // Duplicate the row once so the loop is seamless; clones are hidden from screen readers.
    Array.from(s.children).forEach((item) => {
      const dup = item.cloneNode(true) as HTMLElement;
      dup.setAttribute("aria-hidden", "true");
      dup.dataset.clone = "true";
      dup.querySelectorAll("a").forEach((a) => a.setAttribute("tabindex", "-1"));
      s.appendChild(dup);
    });
    c.style.setProperty("--animation-direction", direction === "left" ? "forwards" : "reverse");
    c.style.setProperty("--animation-duration", speed === "fast" ? "20s" : speed === "normal" ? "40s" : "120s");
    setStart(true);
  }, [direction, speed]);

  // Taking over by hand: stop the marquee, drop the clones, keep the current position.
  const takeOver = () => {
    const s = scrollerRef.current;
    const c = containerRef.current;
    if (!s || !c || manual) return;
    // The marquee never passes -50% (the clone half), so the visible spot is within the originals.
    const x = -new DOMMatrix(getComputedStyle(s).transform).m41;
    s.querySelectorAll("[data-clone]").forEach((n) => n.remove());
    setManual(true);
    requestAnimationFrame(() => c.scrollTo({ left: x }));
  };

  const page = (dir: 1 | -1) => {
    const wasAuto = !manual;
    takeOver();
    const c = containerRef.current;
    // After a take-over, wait one frame for the position restore before paging.
    const go = () => c?.scrollBy({ left: dir * c.clientWidth * 0.8, behavior: "smooth" });
    if (wasAuto) requestAnimationFrame(() => requestAnimationFrame(go));
    else go();
  };

  return (
    <div className="group/strip relative">
      <div
        ref={containerRef}
        onTouchStart={takeOver}
        className={cn(
          "scroller relative z-20 overflow-x-auto overscroll-x-contain [scrollbar-width:none] [&::-webkit-scrollbar]:hidden",
          !manual && "[mask-image:linear-gradient(to_right,transparent,white_8%,white_92%,transparent)]",
          manual && "snap-x snap-mandatory scroll-px-4 px-4",
          className,
        )}
      >
        <ul
          ref={scrollerRef}
          className={cn(
            "flex w-max min-w-full shrink-0 flex-nowrap gap-3 py-2",
            start && !manual && "animate-scroll",
            pauseOnHover && !manual && "hover:[animation-play-state:paused]",
          )}
        >
          {items.map((item, idx) => (
            <li key={itemKey(item, idx)} className={cn("relative shrink-0 snap-start", itemClassName)}>
              {renderItem(item)}
            </li>
          ))}
        </ul>
      </div>
      {(
        [
          [-1, "left-2", IconChevronLeft, `Previous ${label}`],
          [1, "right-2", IconChevronRight, `Next ${label}`],
        ] as const
      ).map(([dir, pos, Icon, aria]) => (
        <button
          key={dir}
          type="button"
          onClick={() => page(dir)}
          aria-label={aria}
          className={cn(
            "absolute top-1/2 z-30 grid h-9 w-9 -translate-y-1/2 place-items-center rounded-full border border-neutral-200 bg-white/90 text-neutral-700 shadow-md backdrop-blur transition hover:bg-white focus-visible:ring-2 focus-visible:ring-cyan-500 focus-visible:outline-none dark:border-white/15 dark:bg-[#1c1c1a]/90 dark:text-neutral-100 dark:hover:bg-[#26261f]",
            pos,
          )}
        >
          <Icon className="h-4 w-4" />
        </button>
      ))}
    </div>
  );
}

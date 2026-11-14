"use client";

import type { ReactNode } from "react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

/**
 * Shared building blocks for the Folio landing page. Every section uses these
 * so buttons, type, spacing and surfaces stay the same from top to bottom.
 */

/** Page-wide palette tokens (light + dark), applied once on the page root. */
export const FOLIO_THEME =
  "[--folio-blue:#4b6bfb] [--folio-card:#ffffff] [--folio-ink:#1a1a1a] [--folio-line:#cfcfc8] [--folio-muted:#5f5f5b] [--folio-paper:#fafaf7] selection:bg-(--folio-blue)/15 dark:[--folio-card:#1c1c1a] dark:[--folio-ink:#f4f4f0] dark:[--folio-line:#3f3f3a] dark:[--folio-muted:#a8a89f] dark:[--folio-paper:#0e0e0c]";

export const surfaces = {
  plain: "bg-white dark:bg-neutral-950",
  tinted: "bg-[#fafaf7] dark:bg-[#0e0e0c]",
} as const;

export const text = {
  h3: "text-lg font-medium tracking-tight text-neutral-900 dark:text-white",
  body: "text-sm/6 text-neutral-600 dark:text-neutral-400",
  lead: "text-base text-pretty text-neutral-600 sm:text-lg dark:text-neutral-400",
} as const;

export function Section({
  id,
  tone = "plain",
  className,
  children,
}: {
  id?: string;
  tone?: keyof typeof surfaces;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section
      id={id}
      className={cn(
        "w-full scroll-mt-16 px-4 py-20 sm:px-6 md:py-28 lg:px-8",
        surfaces[tone],
        className,
      )}
    >
      <div className="mx-auto w-full max-w-7xl">{children}</div>
    </section>
  );
}

export function SectionHeader({
  title,
  description,
  action,
  align = "split",
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  /** `split` puts the description beside the title on large screens. */
  align?: "split" | "center" | "stack";
  className?: string;
}) {
  const heading = (
    <h2 className="text-3xl font-medium tracking-tight text-balance text-neutral-900 sm:text-4xl lg:text-5xl dark:text-white">
      {title}
    </h2>
  );

  if (align === "split") {
    return (
      <div
        className={cn(
          "flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between lg:gap-16",
          className,
        )}
      >
        <div className="max-w-2xl">{heading}</div>
        {description || action ? (
          <div className="flex max-w-md flex-col items-start gap-6">
            {description ? <p className={text.lead}>{description}</p> : null}
            {action}
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex max-w-2xl flex-col gap-5",
        align === "center" && "mx-auto items-center text-center",
        className,
      )}
    >
      {heading}
      {description ? <p className={text.lead}>{description}</p> : null}
      {action}
    </div>
  );
}

const PRESS = { scale: 0.96 } as const;
const PRESS_TRANSITION = { duration: 0.12, ease: [0.23, 1, 0.32, 1] } as const;

const buttonVariants = {
  solid:
    "bg-[#1a1a1a] text-white hover:bg-black dark:bg-[#f4f4f0] dark:text-[#1c1c1a] dark:hover:bg-white",
  outline:
    "bg-white text-[#1a1a1a] ring-1 ring-black/10 hover:bg-neutral-50 dark:bg-[#1c1c1a] dark:text-[#f4f4f0] dark:ring-white/12 dark:hover:bg-neutral-900",
} as const;

export function Button({
  href,
  variant = "solid",
  className,
  children,
}: {
  href: string;
  variant?: keyof typeof buttonVariants;
  className?: string;
  children: ReactNode;
}) {
  // Same markup on server and client: whileTap is always set (a 4% press, not a motion hazard).
  return (
    <motion.a
      href={href}
      whileTap={PRESS}
      transition={PRESS_TRANSITION}
      className={cn(
        "relative inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full px-4 py-2 text-base font-medium sm:min-h-9 sm:px-3.5 sm:py-1.5 sm:text-sm",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#1a1a1a] dark:focus-visible:outline-[#f4f4f0]",
        buttonVariants[variant],
        className,
      )}
    >
      {children}
      <span
        aria-hidden="true"
        className="absolute top-1/2 left-1/2 size-[max(100%,2.75rem)] -translate-1/2 pointer-fine:hidden"
      />
    </motion.a>
  );
}

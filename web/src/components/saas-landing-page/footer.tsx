"use client";

import { motion, useReducedMotion } from "motion/react";
import { OceanMark } from "@/components/brand/ocean-mark";

/** Aceternity footer, adapted: only real links (page sections), the disclaimer, ODG branding. */
export function Footer({ links, note }: { links: { href: string; label: string }[]; note: string }) {
  const reduceMotion = useReducedMotion();

  return (
    <footer className="relative overflow-hidden bg-white px-4 pt-16 pb-10 sm:px-6 lg:px-8 dark:bg-neutral-950">
      <div className="mx-auto max-w-7xl">
        <div className="grid grid-cols-1 gap-10 md:grid-cols-[2fr_1fr]">
          <div>
            <a href="#top" className="flex w-fit items-center gap-2">
              <OceanMark className="size-8 text-neutral-900 dark:text-neutral-100" />
              <span className="text-lg font-medium tracking-tight text-neutral-900 dark:text-neutral-100">StockSentinel</span>
            </a>
            <p className="mt-4 max-w-sm text-sm text-neutral-600 dark:text-neutral-400">
              Daily stock picks from market prices, news and market mood, with every result published. By Ocean Data Group.
            </p>
            <p className="mt-4 max-w-sm text-xs text-neutral-500">{note}</p>
          </div>
          <div>
            <h3 className="text-sm font-medium text-neutral-900 dark:text-neutral-100">On this page</h3>
            <ul className="mt-4 space-y-2">
              {links.map((link) => (
                <li key={link.href}>
                  <a href={link.href} className="text-sm text-neutral-600 transition-colors hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100">
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-12 border-t border-neutral-200 pt-8 dark:border-neutral-800">
          <p className="text-center text-sm text-neutral-500 dark:text-neutral-400">© {new Date().getFullYear()} Ocean Data Group</p>
        </div>
      </div>

      <div
        aria-hidden
        className="pointer-events-none relative mx-auto -mb-[9%] flex max-w-[1180px] items-center justify-center px-4 pb-2 text-center text-[3.5rem] leading-none font-bold sm:text-[7rem] md:text-[8rem] lg:text-[10rem] xl:text-[11rem]"
      >
        <motion.div
          animate={reduceMotion ? undefined : { opacity: [1, 0.55, 1] }}
          transition={{ duration: 4, ease: "easeInOut", repeat: Infinity }}
          className="text-white drop-shadow-xl drop-shadow-black/10 dark:text-neutral-950 dark:drop-shadow-white/10"
        >
          StockSentinel
        </motion.div>
        <div className="absolute bottom-0 left-0 z-20 h-[20%] w-full bg-linear-to-b from-transparent via-white to-white dark:via-neutral-950 dark:to-neutral-950" />
      </div>
    </footer>
  );
}

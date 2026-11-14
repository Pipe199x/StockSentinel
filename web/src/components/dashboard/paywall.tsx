"use client";

/**
 * Paywall UI. The real protection is server-side (functions/data/_middleware.ts serves the
 * premium files only with a session); these pieces only present it:
 * - <Locked> blurs *placeholder* text (never real data) and offers the unlock dialog;
 * - <UnlockDialog> sends a Gumroad license key to /api/unlock.
 */
import { createContext, useContext, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "motion/react";
import { IconLock, IconX } from "@tabler/icons-react";
import { cn } from "@/lib/utils";

export const GUMROAD_URL = "https://gumroad.com/l/qgbjg";

type Paywall = { locked: boolean; openUnlock: () => void };
export const PaywallContext = createContext<Paywall>({ locked: false, openUnlock: () => {} });
export const usePaywall = () => useContext(PaywallContext);

/** Blurred stand-in text. `children` must be placeholder content when locked. */
export function Locked({ children, className, badge = false }: { children: React.ReactNode; className?: string; badge?: boolean }) {
  const { locked } = usePaywall();
  if (!locked) return <>{children}</>;
  return (
    <span className={cn("relative inline-block", className)}>
      <span aria-hidden className="pointer-events-none blur-[5px] select-none">
        {children}
      </span>
      <span className="sr-only">Locked</span>
      {/* A label, not a button: it sits inside clickable cards, which open the dialog themselves. */}
      {badge && (
        <span aria-hidden className="absolute inset-0 m-auto flex h-7 w-fit items-center gap-1 rounded-full bg-neutral-900 px-2.5 text-[11px] font-medium whitespace-nowrap text-white shadow-md dark:bg-white dark:text-neutral-900">
          <IconLock className="h-3 w-3" /> Locked
        </span>
      )}
    </span>
  );
}

/** A whole block (tabs, FAQ answer) behind a blur with one unlock call to action. */
export function LockedBlock({ children, label = "Unlock to read" }: { children: React.ReactNode; label?: string }) {
  const { openUnlock } = usePaywall();
  return (
    <div className="relative">
      <div aria-hidden className="pointer-events-none blur-[6px] select-none">
        {children}
      </div>
      <div className="absolute inset-0 grid place-items-center">
        <button
          type="button"
          onClick={openUnlock}
          className="flex items-center gap-1.5 rounded-full bg-neutral-900 px-4 py-2 text-sm font-medium text-white shadow-lg focus-visible:outline-2 focus-visible:outline-offset-2 dark:bg-white dark:text-neutral-900"
        >
          <IconLock className="h-4 w-4" /> {label}
        </button>
      </div>
    </div>
  );
}

export function UnlockDialog({ open, onClose, onUnlock }: { open: boolean; onClose: () => void; onUnlock: (key: string) => Promise<string | null> }) {
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [portal, setPortal] = useState(false);
  useEffect(() => setPortal(true), []);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!key.trim()) return;
    setBusy(true);
    setErr(null);
    const problem = await onUnlock(key.trim());
    setBusy(false);
    if (problem) setErr(problem);
    else {
      setKey("");
      onClose();
    }
  };

  if (!portal) return null;
  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[1100] grid place-items-center bg-black/50 p-4 backdrop-blur-sm" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="unlock-title"
            initial={{ y: 16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 8, opacity: 0 }}
            className="w-full max-w-md rounded-3xl bg-white p-6 text-neutral-900 ring-1 ring-black/10 dark:bg-[#1c1c1a] dark:text-neutral-100 dark:ring-white/10"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id="unlock-title" className="text-xl font-semibold tracking-tight">
                  Unlock today&apos;s picks
                </h2>
                <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">Tickers, targets, stops, odds, open positions and the technical details.</p>
              </div>
              <button type="button" onClick={onClose} aria-label="Close" className="rounded-full p-1.5 hover:bg-black/5 dark:hover:bg-white/10">
                <IconX className="h-4 w-4" />
              </button>
            </div>
            <form onSubmit={submit} className="mt-5 space-y-3">
              <label className="block text-sm font-medium" htmlFor="license-key">
                License key from your Gumroad receipt
              </label>
              <input
                id="license-key"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                autoComplete="off"
                spellCheck={false}
                placeholder="XXXXXXXX-XXXXXXXX-XXXXXXXX-XXXXXXXX"
                className="w-full rounded-xl bg-neutral-100 px-3 py-2.5 font-mono text-sm ring-1 ring-black/10 outline-none focus:ring-2 focus:ring-[#4b6bfb] dark:bg-white/[0.06] dark:ring-white/10"
              />
              {err && (
                <p role="alert" className="text-sm text-rose-600 dark:text-rose-400">
                  {err}
                </p>
              )}
              <button
                type="submit"
                disabled={busy || !key.trim()}
                className="w-full rounded-full bg-neutral-900 px-4 py-2.5 text-sm font-medium text-white disabled:opacity-50 dark:bg-white dark:text-neutral-900"
              >
                {busy ? "Checking…" : "Unlock"}
              </button>
            </form>
            <p className="mt-4 text-center text-sm text-neutral-600 dark:text-neutral-400">
              No key yet?{" "}
              <a href={GUMROAD_URL} target="_blank" rel="noopener noreferrer" className="font-medium text-[#4b6bfb] underline-offset-4 hover:underline">
                Get access on Gumroad
              </a>
            </p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  );
}

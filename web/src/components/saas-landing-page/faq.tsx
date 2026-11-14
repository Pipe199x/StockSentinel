"use client";

import { useId, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { IconPlus } from "@tabler/icons-react";
import { cn } from "@/lib/utils";
import {
  Section,
  SectionHeader,
  text,
} from "@/components/saas-landing-page/ui";

export type Question = {
  question: string;
  answer: string;
};

export function FAQ({
  heading,
  subheading,
  questions,
  children,
  className,
  locked = false,
  onUnlock,
}: {
  heading: string;
  subheading?: string;
  questions: Question[];
  /** Paywall: answers are shown blurred with an unlock button. */
  locked?: boolean;
  onUnlock?: () => void;
  /** Extra content under the header column (e.g. the technical details). */
  children?: React.ReactNode;
  className?: string;
}) {
  const [open, setOpen] = useState<number | null>(0);

  return (
    <Section id="faq" className={className}>
      <div className="grid grid-cols-1 gap-12 lg:grid-cols-[1fr_1.4fr] lg:gap-20">
        <div>
          <SectionHeader align="stack" title={heading} description={subheading} />
          {children}
        </div>
        <ul className="divide-y divide-neutral-200 border-y border-neutral-200 dark:divide-neutral-800 dark:border-neutral-800">
          {questions.map((item, index) => (
            <FAQItem
              key={item.question}
              item={item}
              isOpen={open === index}
              onToggle={() => setOpen(open === index ? null : index)}
              locked={locked}
              onUnlock={onUnlock}
            />
          ))}
        </ul>
      </div>
    </Section>
  );
}

function FAQItem({
  item,
  isOpen,
  onToggle,
  locked,
  onUnlock,
}: {
  item: Question;
  isOpen: boolean;
  onToggle: () => void;
  locked?: boolean;
  onUnlock?: () => void;
}) {
  const id = useId();
  const reduceMotion = useReducedMotion();

  return (
    <li>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isOpen}
        aria-controls={id}
        className="flex w-full items-center justify-between gap-6 py-5 text-left"
      >
        <span className="text-base font-medium tracking-tight text-neutral-900 dark:text-white">
          {item.question}
        </span>
        <motion.span
          animate={{ rotate: isOpen ? 45 : 0 }}
          transition={{
            duration: reduceMotion ? 0 : 0.2,
            ease: [0.23, 1, 0.32, 1],
          }}
          className="flex size-6 shrink-0 items-center justify-center rounded-full bg-neutral-100 text-neutral-600 dark:bg-neutral-800 dark:text-neutral-300"
        >
          <IconPlus className="size-3.5" />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            id={id}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{
              duration: reduceMotion ? 0 : 0.25,
              ease: [0.23, 1, 0.32, 1],
            }}
            className="overflow-hidden"
          >
            {locked ? (
              <div className="relative pb-5">
                <p aria-hidden className={cn(text.body, "pointer-events-none max-w-prose blur-[5px] select-none")}>
                  {item.answer}
                </p>
                <button
                  type="button"
                  onClick={onUnlock}
                  className="absolute top-1 left-0 rounded-full bg-neutral-900 px-3 py-1.5 text-xs font-medium text-white shadow-md dark:bg-white dark:text-neutral-900"
                >
                  Unlock to read the answer
                </button>
              </div>
            ) : (
              <p className={cn(text.body, "max-w-prose pb-5")}>{item.answer}</p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}

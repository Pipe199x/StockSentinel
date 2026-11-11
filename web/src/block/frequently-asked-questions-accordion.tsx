"use client";
import React, { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { IconMinus, IconPlus } from "@tabler/icons-react";
import { cn } from "@/lib/utils";

export type FAQ = { question: string; answer: React.ReactNode };

/** Aceternity "Frequently Asked Questions Accordion", adapted to take content and fit a card. */
export default function FrequentlyAskedQuestionsAccordion({ title, faqs }: { title: string; faqs: FAQ[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div>
      <h3 className="text-sm font-semibold text-neutral-700 dark:text-neutral-100">{title}</h3>
      <div className="divide-y divide-neutral-200 dark:divide-neutral-800">
        {faqs.map((faq) => (
          <FAQItem key={faq.question} question={faq.question} answer={faq.answer} open={open} setOpen={setOpen} />
        ))}
      </div>
    </div>
  );
}

const FAQItem = ({
  question,
  answer,
  setOpen,
  open,
}: {
  question: string;
  answer: React.ReactNode;
  open: string | null;
  setOpen: (open: string | null) => void;
}) => {
  const isOpen = open === question;

  return (
    <div
      className="cursor-pointer py-3"
      onClick={() => {
        if (isOpen) {
          setOpen(null);
        } else {
          setOpen(question);
        }
      }}
    >
      <div className="flex items-start">
        <div className="relative mr-3 mt-0.5 h-5 w-5 flex-shrink-0">
          <IconPlus
            className={cn(
              "absolute inset-0 h-5 w-5 transform text-cyan-500 transition-all duration-200",
              isOpen && "rotate-90 scale-0",
            )}
          />
          <IconMinus
            className={cn(
              "absolute inset-0 h-5 w-5 rotate-90 scale-0 transform text-cyan-500 transition-all duration-200",
              isOpen && "rotate-0 scale-100",
            )}
          />
        </div>
        <div>
          <h4 className="text-sm font-medium text-neutral-700 dark:text-neutral-200">
            {question}
          </h4>
          <AnimatePresence mode="wait">
            {isOpen && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: "auto" }}
                exit={{ height: 0 }}
                transition={{ duration: 0.2, ease: "easeOut" }}
                className="overflow-hidden pt-1 text-sm text-neutral-500 dark:text-neutral-400"
              >
                <p>{answer}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

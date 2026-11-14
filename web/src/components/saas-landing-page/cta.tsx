import type { ReactNode } from "react";
import { Button, Section, surfaces, text } from "@/components/saas-landing-page/ui";
import { cn } from "@/lib/utils";

/** Aceternity CTA band, adapted: no testimonial figure; an optional fact panel instead. */
export function CTA({
  title,
  lead,
  primary,
  secondary,
  aside,
}: {
  title: string;
  lead: string;
  primary: { href: string; label: string };
  secondary: { href: string; label: string };
  aside?: ReactNode;
}) {
  return (
    <Section id="cta" className="pt-0 md:pt-0">
      <div className={cn("grid grid-cols-1 overflow-hidden rounded-3xl ring-1 ring-black/8 md:grid-cols-3 dark:ring-white/10", surfaces.tinted)}>
        <div className={cn("flex flex-col items-start gap-5 p-8 md:p-12", aside ? "md:col-span-2" : "md:col-span-3")}>
          <h2 className="text-3xl font-medium tracking-tight text-balance text-neutral-900 sm:text-4xl dark:text-white">{title}</h2>
          <p className={cn(text.lead, "max-w-lg")}>{lead}</p>
          <div className="mt-2 flex flex-wrap items-center gap-2.5">
            <Button href={primary.href}>{primary.label}</Button>
            <Button href={secondary.href} variant="outline">
              {secondary.label}
            </Button>
          </div>
        </div>
        {aside && <div className="flex flex-col justify-center gap-4 border-t border-black/8 p-8 md:border-t-0 md:border-l md:p-12 dark:border-white/10">{aside}</div>}
      </div>
    </Section>
  );
}

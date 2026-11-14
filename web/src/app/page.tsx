"use client";

/**
 * StockSentinel on the Aceternity "SaaS landing page" template:
 * Hero (beam diagram) → Today's picks → Latest headlines → the story → Model portfolio → FAQ →
 * Technical details → CTA → Footer. Every number comes from the published JSON (docs/data).
 */
import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { ReactLenis } from "lenis/react";
import { IconActivity, IconChartLine, IconLock, IconMoon, IconNews, IconShieldHalf, IconSun } from "@tabler/icons-react";
import { cn } from "@/lib/utils";
import { LoaderTwo } from "@/components/ui/loader";
import { Watermark } from "@/components/brand/watermark";
import { Hero, ODG_URL, type HeroNode } from "@/components/saas-landing-page/hero";
import { FOLIO_THEME, Section, SectionHeader } from "@/components/saas-landing-page/ui";
import { FAQ, type Question } from "@/components/saas-landing-page/faq";
import { CTA } from "@/components/saas-landing-page/cta";
import { Footer } from "@/components/saas-landing-page/footer";
import { heroCopy, HeroEyebrow, HeroFacts } from "@/components/dashboard/verdict";
import { HeadlineStrip } from "@/components/dashboard/headlines";
import { TradeCards } from "@/components/dashboard/trade-cards";
import { TrackRecord } from "@/components/dashboard/track-record";
import { TechnicalDetails } from "@/components/dashboard/trust";
import { ACCOUNT, money, Story } from "@/components/dashboard/story";
import { useDashboard } from "@/lib/data";
import { PaywallContext, UnlockDialog } from "@/components/dashboard/paywall";
import { longDate, num } from "@/lib/format";
import { diffVisit, readSnapshot, takeSnapshot, writeSnapshot, type VisitDiff } from "@/lib/last-visit";

const LINKS = [
  { href: "#trades", label: "Today's picks" },
  { href: "#story", label: "The story" },
  { href: "#track", label: "Model portfolio" },
  { href: "#faq", label: "FAQ" },
];
const NOTE = "Model portfolio; results are hypothetical. Not advice.";

const iconBtn =
  "grid size-9 place-items-center rounded-full text-(--folio-ink) ring-1 ring-black/10 transition hover:bg-black/[0.04] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-(--folio-ink) dark:ring-white/12 dark:hover:bg-white/[0.06]";

function ThemeToggle() {
  const [dark, setDark] = useState(false);
  useEffect(() => setDark(document.documentElement.classList.contains("dark")), []);
  const flip = () => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {}
    setDark(next);
  };
  return (
    <button type="button" onClick={flip} aria-label={dark ? "Switch to light mode" : "Switch to dark mode"} className={iconBtn}>
      {dark ? <IconSun className="h-4 w-4" /> : <IconMoon className="h-4 w-4" />}
    </button>
  );
}

/** Sections rise into place as they scroll into view. */
function Reveal({ children }: { children: React.ReactNode }) {
  return (
    <motion.div initial={{ opacity: 0, y: 32 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-80px" }} transition={{ duration: 0.6, ease: "easeOut" }}>
      {children}
    </motion.div>
  );
}

const INPUTS: HeroNode[] = [
  { key: "prices", label: "Prices for 25 assets", content: <IconChartLine className="size-5" /> },
  { key: "news", label: "Financial news", content: <IconNews className="size-5" /> },
  { key: "mood", label: "Market mood", content: <IconActivity className="size-5" /> },
  { key: "risk", label: "Risk limits", content: <IconShieldHalf className="size-5" /> },
];

export default function Home() {
  const { data, error, loading, unlock, logout } = useDashboard();
  const [visit, setVisit] = useState<VisitDiff | null>(null);
  const [unlockOpen, setUnlockOpen] = useState(false);
  const locked = data?.locked ?? true;
  const paywall = { locked, openUnlock: () => setUnlockOpen(true) };
  const reduced = useReducedMotion();

  // Diff against the previous visit once per page load, then remember what was seen.
  useEffect(() => {
    if (!data || visit) return;
    const closed = data.paper?.recent_closed ?? [];
    setVisit(diffVisit(readSnapshot(), data.signals, closed));
    writeSnapshot(takeSnapshot(data.signals, closed));
  }, [data, visit]);

  const signals = data?.signals ?? [];
  const copy = heroCopy(signals, locked);
  const idle = signals.filter((s) => !copy.active.includes(s)).sort((a, b) => a.ticker.localeCompare(b.ticker));
  // Right side of the diagram: today's picks first, then (dimmed) assets that got no pick.
  const outputs: HeroNode[] = [
    ...copy.active.map((s, i) => ({
      key: s.ticker,
      label: locked ? `Pick ${i + 1}: locked` : `${s.ticker}: ${s.side === 1 ? "long" : "short"} pick today`,
      href: "#trades",
      content: (
        <span className={s.side === 1 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>
          {locked ? <span aria-hidden className="blur-[4px] select-none">XXX</span> : s.ticker}
        </span>
      ),
    })),
    ...idle.map((s) => ({
      key: s.ticker,
      label: locked ? "Asset with no pick today" : `${s.ticker}: no pick today`,
      href: "#trades",
      muted: true,
      content: locked ? <span aria-hidden className="blur-[4px] select-none">XXX</span> : s.ticker,
    })),
  ].slice(0, 4);

  const r = data?.report;
  const faq: Question[] = [
    {
      question: "What is StockSentinel?",
      answer: `A system that reads market prices for ${signals.length || 25} assets, financial news and the mood of the overall market every market day, and publishes the trades its rules select. Ocean Data Group builds and runs it.`,
    },
    {
      question: "How does it pick?",
      answer:
        "First it checks the market mood; in a stress period it does not trade. Then a momentum rule looks for assets with a trend, and a second model estimates the chance each trade reaches its profit target before its stop. Only trades where that chance is high enough get a size.",
    },
    {
      question: `What is the ${money(ACCOUNT)} model portfolio?`,
      answer: `An account that starts at ${money(ACCOUNT)} and follows every pick with the same rules: it buys at the next day's open, sells at the target, the stop or the exit date, and pays 0.10% in costs per trade. All figures on this page are for that account.`,
    },
    {
      question: "Why is most of the money in cash?",
      answer:
        "Each pick gets a small share of the account, sized so that one bad trade costs at most about 1% of the account. With a few picks open at a time, most of the account stays in cash.",
    },
    {
      question: "How was it tested?",
      answer: r
        ? `On ${num(r.meta.years, 1)} years of market history it had never seen while it was built: ${r.meta.n_trades.toLocaleString()} trades in total. The chance that the results come from skill rather than luck is ${Math.round((r.meta.dsr ?? 0) * 100)}%; researchers use ${Math.round(r.dsr_threshold * 100)}% as the line. Since ${data?.paper?.started ? longDate(data.paper.started) : "launch"} the model portfolio also follows the picks as they happen.`
        : "On years of market history it had never seen while it was built, and from launch on in a model portfolio that follows the picks as they happen.",
    },
    { question: "Is this advice?", answer: "No. The picks and the model portfolio show what StockSentinel's rules do. They are not a recommendation to buy or sell anything." },
  ];

  const page = (
    <div className={cn("relative isolate min-h-screen w-full overflow-x-clip bg-(--folio-paper) text-(--folio-ink)", FOLIO_THEME)}>
      <Watermark variant="corner" fixed position="bottom-right" opacity={0.5} href={ODG_URL} className="z-50 origin-bottom-right scale-75 sm:scale-100" />

      <Hero
        links={LINKS}
        actions={
          <>
            <ThemeToggle />
            {data &&
              (locked ? (
                <button type="button" onClick={paywall.openUnlock} className="flex h-9 items-center gap-1.5 rounded-full bg-(--folio-ink) px-3.5 text-sm font-medium text-(--folio-paper)">
                  <IconLock className="h-4 w-4" /> Unlock
                </button>
              ) : (
                <button type="button" onClick={logout} className="flex h-9 items-center rounded-full px-3.5 text-sm text-(--folio-muted) ring-1 ring-black/10 hover:text-(--folio-ink) dark:ring-white/12">
                  Sign out
                </button>
              ))}
          </>
        }
        eyebrow={data && <HeroEyebrow signals={signals} report={r ?? null} />}
        title={loading && !data ? "Loading today's picks…" : copy.title}
        lead={error ? error : copy.lead}
        primary={{ href: "#trades", label: "See today's picks" }}
        secondary={{ href: "#story", label: `What did ${money(ACCOUNT)} become?` }}
        extra={
          <>
            {loading && !data && <LoaderTwo />}
            {data && <HeroFacts report={r ?? null} visit={visit} />}
          </>
        }
        inputs={INPUTS}
        outputs={outputs}
        labels={{ inPill: "Every market day", outPill: "Today's picks", inNote: "Prices, news, market mood", outNote: copy.active.length ? `${copy.active.length} pick${copy.active.length > 1 ? "s" : ""} · Not advice` : "No picks today" }}
        caption="Market prices, news and market mood flow into StockSentinel; today's picks come out the other side."
      />

      {data && (
        <main>
          <Section id="trades" tone="tinted">
            <Reveal>
              <SectionHeader title="Today's picks" description="Tap a pick for its chart, the reasons behind it, the risk and its news. Not advice." />
              <div className="mt-10">
                <TradeCards
                  signals={signals}
                  prices={data.prices}
                  news={data.news}
                  newTickers={visit?.newSignals ?? []}
                  universe={Object.values(data.universe?.sectors ?? {}).flat().sort()}
                />
              </div>
            </Reveal>
          </Section>

          <Section id="headlines">
            <SectionHeader title="Latest headlines" description="The news StockSentinel reads, newest first. Each dot shows the tone of the article." />
            <div className="mt-10">
              <HeadlineStrip news={data.news} />
            </div>
          </Section>

          <Section id="story" tone="tinted">
            <SectionHeader title={`${money(ACCOUNT)} following StockSentinel`} description="Every trade on years of market history it had never seen, applied to one account." />
            <div className="mt-6">
              <Story report={data.report} />
            </div>
          </Section>

          <Section id="track">
            <Reveal>
              <SectionHeader title={`StockSentinel's ${money(ACCOUNT)} model portfolio`} description="The same picks, followed as they happen. Not advice." />
              <div className="mt-10">
                <TrackRecord paper={data.paper} visit={visit} report={data.report} />
              </div>
            </Reveal>
          </Section>

          <FAQ heading="Questions, answered" subheading="What people ask before they read the picks." questions={faq} locked={locked} onUnlock={paywall.openUnlock} className="bg-[#fafaf7] dark:bg-[#0e0e0c]" />

          <Section id="details" className="pt-0 md:pt-0">
            <SectionHeader align="stack" title="Technical details" description="The statistics behind the story, for people who want them." />
            <div className="mt-8">
              <TechnicalDetails report={data.report} risk={data.risk} deep={data.deep} locked={locked} />
            </div>
          </Section>

          <CTA
            title="New picks every market day"
            lead="StockSentinel publishes its picks each market day, and the model portfolio updates with them."
            primary={{ href: "#trades", label: "See today's picks" }}
            secondary={{ href: "#story", label: `Read the ${money(ACCOUNT)} story` }}
          />
        </main>
      )}

      <Footer links={LINKS} note={NOTE} />
    </div>
  );

  const gated = (
    <PaywallContext.Provider value={paywall}>
      {page}
      <UnlockDialog open={unlockOpen} onClose={() => setUnlockOpen(false)} onUnlock={unlock} />
    </PaywallContext.Provider>
  );

  return reduced ? gated : <ReactLenis root options={{ anchors: { offset: -72 }, lerp: 0.1 }}>{gated}</ReactLenis>;
}

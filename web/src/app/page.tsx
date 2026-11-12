"use client";

import { useEffect, useMemo, useState } from "react";
import { IconMoon, IconRefresh, IconSun } from "@tabler/icons-react";
import { Navbar, NavBody, NavItems, MobileNav, MobileNavHeader, MobileNavMenu, MobileNavToggle, NavbarLogo, NavbarButton } from "@/components/ui/resizable-navbar";
import { GridWithDots } from "@/block/background-grid-with-dots";
import StatsWithNumberTicker, { type StatItem } from "@/block/stats-with-number-ticker";
import { CardSpotlight } from "@/components/ui/card-spotlight";
import { LoaderTwo } from "@/components/ui/loader";
import { Watermark } from "@/components/brand/watermark";
import { LinesGradientShader } from "@/block/lines-gradient-shader";
import { SignalsBoard } from "@/components/dashboard/signals-board";
import { AssetDetail } from "@/components/dashboard/asset-detail";
import { Validation } from "@/components/dashboard/validation";
import { Paper } from "@/components/dashboard/paper";
import { Models, NewsFeed } from "@/components/dashboard/models-news";
import { useDashboard } from "@/lib/data";
import { longDate, num, REGIME } from "@/lib/format";

const NAV = [
  { name: "Signals", link: "#signals" },
  { name: "Asset", link: "#asset" },
  { name: "Validation", link: "#validation" },
  { name: "Paper", link: "#paper" },
  { name: "Models", link: "#models" },
  { name: "News", link: "#news" },
];

/** True on phone-width screens; drives lighter shader settings. */
function useIsPhone() {
  const [phone, setPhone] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 640px)");
    const sync = () => setPhone(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);
  return phone;
}

function ThemeToggle() {
  const [dark, setDark] = useState(true);
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
    <NavbarButton as="button" variant="secondary" onClick={flip} aria-label={dark ? "Switch to light mode" : "Switch to dark mode"} className="px-2">
      {dark ? <IconSun className="h-4 w-4" /> : <IconMoon className="h-4 w-4" />}
    </NavbarButton>
  );
}

export default function Home() {
  const { data, error, loading, reload } = useDashboard();
  const [ticker, setTicker] = useState<string | null>(null);
  const [menu, setMenu] = useState(false);
  const phone = useIsPhone();

  const signals = data?.signals ?? [];
  useEffect(() => {
    if (!ticker && signals.length) {
      const best = [...signals].sort((a, b) => (b.kelly_size || 0) - (a.kelly_size || 0))[0];
      setTicker(best.ticker);
    }
  }, [signals, ticker]);
  const selected = signals.find((s) => s.ticker === ticker) ?? null;

  const kpis: StatItem[] = useMemo(() => {
    const r = data?.report;
    const p = data?.paper;
    const mkt = signals[0]?.market_regime ?? null;
    const active = signals.filter((s) => s.kelly_size > 0).length;
    return [
      {
        label: "Market regime",
        value: null,
        text: REGIME[mkt ?? ""] ?? "—",
        tone: mkt === "stress" ? "bad" : mkt === "calm" ? "good" : "neutral",
        description: `HMM on SPY, P(stress) ${num((signals[0]?.market_p_stress ?? 0) * 100, 0)}%. Stress means stay flat.`,
      },
      {
        label: "Active signals",
        value: active,
        format: (v) => Math.round(v).toString(),
        description: `${active} of ${signals.length} assets have a non-zero size today.`,
      },
      {
        label: "Deflated Sharpe",
        value: r?.meta.dsr ?? null,
        format: (v) => v.toFixed(3),
        tone: r?.edge_is_significant ? "good" : "neutral",
        description: r?.edge_is_significant ? "Above the 0.95 bar: the edge is unlikely to be luck." : "Below the 0.95 bar: the edge is not proven yet.",
      },
      {
        label: "Paper trading",
        value: p ? p.total_return * 100 : null,
        format: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`,
        tone: p ? (p.total_return >= 0 ? "good" : "bad") : "neutral",
        description: p ? `Gate 0: ${p.gate0.status}. ${p.counts.closed}/${p.gate0.min_trades} closed trades.` : "Starts with the next pipeline run.",
      },
    ];
  }, [data, signals]);

  return (
    <div id="top" className="relative isolate min-h-screen overflow-x-clip">
      {/* Aceternity background + two Ocean Data Group watermarks (pointer-events-none, aria-hidden) */}
      <div className="fixed inset-0 -z-10">
        <GridWithDots className="[mask-image:radial-gradient(ellipse_at_top,black_30%,transparent_80%)]" />
      </div>
      <Watermark variant="center" fixed opacity={0.04} label={false} className="z-0" />
      <Watermark variant="corner" fixed position="bottom-right" opacity={phone ? 0.4 : 0.6} className="z-50 origin-bottom-right scale-75 sm:scale-100" />

      <Navbar>
        <NavBody>
          <NavbarLogo />
          <NavItems items={NAV} />
          <div className="relative z-20 flex items-center gap-2">
            <ThemeToggle />
            <NavbarButton as="button" variant="dark" onClick={reload} className="flex items-center gap-1.5">
              <IconRefresh className="h-4 w-4" /> Refresh
            </NavbarButton>
          </div>
        </NavBody>
        <MobileNav>
          <MobileNavHeader>
            <NavbarLogo />
            <div className="flex items-center gap-2">
              <ThemeToggle />
              <MobileNavToggle isOpen={menu} onClick={() => setMenu(!menu)} />
            </div>
          </MobileNavHeader>
          <MobileNavMenu isOpen={menu} onClose={() => setMenu(false)}>
            {NAV.map((n) => (
              <a key={n.link} href={n.link} onClick={() => setMenu(false)} className="w-full text-neutral-700 dark:text-neutral-200">
                {n.name}
              </a>
            ))}
            <NavbarButton as="button" variant="dark" onClick={() => { setMenu(false); reload(); }} className="w-full">
              Refresh data
            </NavbarButton>
          </MobileNavMenu>
        </MobileNav>
      </Navbar>

      <main className="relative z-10 mx-auto max-w-7xl space-y-6 px-4 pt-10 pb-16 md:px-6">
        <section className="relative isolate pt-4 pb-10">
          <LinesGradientShader
            className="absolute inset-y-0 left-1/2 -z-10 w-screen -translate-x-1/2 bg-white dark:bg-[#0c0d1c]"
            bandCount={phone ? 4 : 5}
            bandSpacing={40}
            bandThickness={phone ? 70 : 100}
            waveAmplitude={0.2}
            speed={1}
            targetFps={phone ? 24 : 30}
          />
          {/* fades the bands into the page so the hero has no hard bottom edge */}
          <div aria-hidden className="pointer-events-none absolute bottom-0 left-1/2 -z-10 h-24 w-screen -translate-x-1/2 bg-gradient-to-b from-transparent to-white dark:to-[#0c0d1c]" />
        <header className="pt-4">
          <p className="text-xs font-semibold tracking-[0.2em] text-cyan-700 uppercase dark:text-cyan-300/80">Quant signals · honest validation</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight md:text-5xl">StockSentinel</h1>
          <p className="mt-3 max-w-2xl text-neutral-600 dark:text-neutral-300">
            Regime-aware signals for {signals.length || 25} assets, validated out-of-sample with purged cross-validation and the Deflated Sharpe, and
            tracked live in a paper portfolio.
            {data?.report && <> Data as of {longDate(data.report.as_of)}.</>}
          </p>
        </header>

        {error && (
          <div role="alert" className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm">
            {error} <button onClick={reload} className="ml-2 underline">Try again</button>
          </div>
        )}

        <CardSpotlight color="#14152b" radius={420} className="mt-6 rounded-2xl border border-neutral-200 bg-white/80 p-4 sm:p-6 md:p-8 dark:border-white/[0.1] dark:bg-[#101124]/80 backdrop-blur-md">
          {loading && !data ? (
            <div className="grid h-24 place-items-center" aria-busy="true" aria-label="Loading data">
              <LoaderTwo />
            </div>
          ) : (
            <StatsWithNumberTicker items={kpis} />
          )}
        </CardSpotlight>
        </section>

        {data && (
          <>
            <div className="space-y-6">
              <SignalsBoard signals={signals} selected={ticker} onSelect={(t) => { setTicker(t); document.getElementById("asset")?.scrollIntoView({ behavior: "smooth", block: "start" }); }} />
              <AssetDetail signal={selected} series={(ticker && data.prices?.series[ticker]) || []} />
            </div>
            <Validation report={data.report} />
            <Paper paper={data.paper} />
            <Models deep={data.deep} />
            <NewsFeed news={data.news} />
          </>
        )}
      </main>

    </div>
  );
}

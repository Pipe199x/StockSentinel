export const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

const nf = (o: Intl.NumberFormatOptions) => new Intl.NumberFormat("en-US", o);
const N2 = nf({ minimumFractionDigits: 2, maximumFractionDigits: 2 });
const N3 = nf({ minimumFractionDigits: 3, maximumFractionDigits: 3 });

export const num = (v: unknown, d = 2) => (fin(v) ? (d === 3 ? N3 : d === 2 ? N2 : nf({ maximumFractionDigits: d })).format(v) : "—");
export const pct = (v: unknown, d = 1) => (fin(v) ? nf({ style: "percent", minimumFractionDigits: d, maximumFractionDigits: d }).format(v) : "—");
export const signedPct = (v: unknown, d = 2) => (fin(v) ? `${v > 0 ? "+" : ""}${pct(v, d)}` : "—");
export const usd = (v: unknown) => (fin(v) ? nf({ style: "currency", currency: "USD" }).format(v) : "—");

export const shortDate = (iso?: string | null) =>
  iso ? new Intl.DateTimeFormat("en-US", { day: "numeric", month: "short", timeZone: "UTC" }).format(new Date(iso.slice(0, 10) + "T00:00:00Z")) : "—";

export const longDate = (iso?: string | null) =>
  iso ? new Intl.DateTimeFormat("en-US", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(iso.slice(0, 10) + "T00:00:00Z")) : "—";

export function ago(iso?: string | null) {
  if (!iso) return "";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  if (s < 3600) return rtf.format(-Math.round(s / 60), "minute");
  if (s < 86400) return rtf.format(-Math.round(s / 3600), "hour");
  return rtf.format(-Math.round(s / 86400), "day");
}

export const SIDE: Record<string, string> = { "1": "Long", "-1": "Short", "0": "Flat" };
export const STRATEGY: Record<string, string> = { none: "No setup", momentum: "Momentum", mean_reversion: "Mean reversion" };
export const REGIME: Record<string, string> = { calm: "Calm", normal: "Normal", stress: "Stress" };
export const SECTOR_EN: Record<string, string> = {
  "Tecnología": "Technology",
  Salud: "Health care",
  "Energía": "Energy",
  Finanzas: "Financials",
  "Consumo básico": "Consumer staples",
  "Consumo discrecional": "Consumer discretionary",
  Industria: "Industrials",
  Utilities: "Utilities",
  ETFs: "ETFs",
};

export function safeUrl(u?: string | null) {
  try {
    const x = new URL(u ?? "");
    return /^https?:$/.test(x.protocol) ? x.href : null;
  } catch {
    return null;
  }
}

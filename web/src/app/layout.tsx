import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "StockSentinel",
  description: "Regime-aware quant signals with honest validation, by Ocean Data Group.",
  icons: { icon: "/brand/odg-mark-navy.svg" },
};

// Light by default (the SaaS template look); the theme toggle stores the choice.
const themeScript = `try{document.documentElement.classList.toggle("dark",localStorage.getItem("theme")==="dark")}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`} suppressHydrationWarning>
      <head>
        {/* suppressHydrationWarning: browser extensions rewrite this tag before React hydrates. */}
        <script suppressHydrationWarning dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-full bg-[#fafaf7] font-sans text-neutral-900 dark:bg-[#0e0e0c] dark:text-neutral-100">
        {children}
      </body>
    </html>
  );
}

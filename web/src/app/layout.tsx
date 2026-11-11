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

// Dark by default (the ODG dashboards are dark); the theme toggle stores the choice.
const themeScript = `try{var t=localStorage.getItem("theme");document.documentElement.classList.toggle("dark",t?t==="dark":true)}catch(e){document.documentElement.classList.add("dark")}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} dark h-full antialiased`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-full bg-white font-sans text-neutral-900 dark:bg-[#0c0d1c] dark:text-neutral-100">
        {children}
      </body>
    </html>
  );
}

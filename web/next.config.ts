import type { NextConfig } from "next";

// Static export: GitHub Pages serves the files from docs/, and the pages fetch
// /data/*.json at runtime, so the daily data commits update the site with no rebuild.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  devIndicators: false,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;

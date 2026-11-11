// Publish the static export into ../docs (GitHub Pages), keeping what the
// pipelines own: docs/data/ (daily JSON), CNAME (custom domain) and .nojekyll.
import { cpSync, existsSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const out = resolve(root, "out");
const docs = resolve(root, "../docs");
const KEEP = new Set(["data", "CNAME", ".nojekyll"]);

if (!existsSync(resolve(out, "index.html"))) throw new Error("web/out/index.html missing: run `next build` first");

for (const name of readdirSync(docs)) if (!KEEP.has(name)) rmSync(resolve(docs, name), { recursive: true, force: true });
for (const name of readdirSync(out)) {
  if (name === "data") continue; // never ship the dev copy over the live data
  cpSync(resolve(out, name), resolve(docs, name), { recursive: true });
}
writeFileSync(resolve(docs, ".nojekyll"), ""); // _next/ must not be filtered by Jekyll
console.log("synced web/out -> docs (kept data/, CNAME, .nojekyll)");

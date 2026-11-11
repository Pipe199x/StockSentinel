// Dev only: copy the published data (../docs/data) into public/data so `next dev`
// serves /data/*.json like GitHub Pages does. public/data is gitignored and the
// production build never ships it (the live site uses docs/data directly).
import { cpSync, rmSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
rmSync(resolve(root, "public/data"), { recursive: true, force: true });
cpSync(resolve(root, "../docs/data"), resolve(root, "public/data"), { recursive: true });
console.log("copied ../docs/data -> public/data");

// Builds the F1 Anatomy site into ./site : an index page and one page per figure.
//   node f1_anatomy/build.mjs            all figures
//   node f1_anatomy/build.mjs corner     one figure (plus the index)
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { FIGURES, figurePage, menuPage } from "./src/lib.mjs";
import * as k from "./vendor/kit/iso-kit.mjs";
import { EXPLAIN, OVERVIEW } from "./src/explain.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = join(HERE, "site");
mkdirSync(join(SITE, "thumbs"), { recursive: true });

const only = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const modules = {};
for (const f of FIGURES) {
  if (only.length && !only.includes(f.id)) continue;
  try {
    modules[f.id] = await import(`./src/figures/${f.id}.mjs`);
  } catch (error) {
    if (error.code === "ERR_MODULE_NOT_FOUND" && String(error.message).includes(`${f.id}.mjs`)) {
      console.log(`skip ${f.id} (not written yet)`);
      continue;
    }
    throw error;
  }
}

const cards = [];
for (const f of FIGURES) {
  const m = modules[f.id];
  if (!m) continue;
  const figure = m.build();
  writeFileSync(join(SITE, `${f.id}.html`), figurePage({ ...figure, explain: EXPLAIN[f.id] }));
  // thumbnails for the index: the figure's own drawing, standalone, in both themes
  for (const theme of ["dark", "light"]) {
    const svg = k.figureSvg({ width: figure.width, height: figure.height, label: figure.title, body: figure.svg, standalone: theme });
    writeFileSync(join(SITE, "thumbs", `${f.id}-${theme}.svg`), svg);
  }
  const first = EXPLAIN[f.id]?.plain?.tryFirst?.[0]?.split("|")[0] || m.meta.tryit;
  cards.push({ id: f.id, num: f.num, title: f.title, blurb: OVERVIEW.how.find((h) => h.name.startsWith(f.num + "."))?.text || m.meta.blurb, tryit: first });
  console.log("built", f.id);
}

// the index lists every figure, built or not, so the menu is complete while figures are added
const all = FIGURES.map((f) => cards.find((c) => c.id === f.id) || { id: f.id, num: f.num, title: f.title, blurb: "Coming next.", tryit: "" });
writeFileSync(join(SITE, "index.html"), menuPage(all, OVERVIEW));
console.log("wrote site/index.html");

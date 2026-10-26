// Screenshot a built page, optionally after driving it.
//   node tools/shot.mjs <page> <out.png> [--hash "speed=300&brake=100"] [--wait 2000] [--w 1100] [--h 720]
//        [--stage] [--zoom x,y,w,h] [--scale 2] [--theme light] [--eval "js"]
// --stage shows only the drawing. --zoom clips a region of the page (CSS px) at --zoom-scale.
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { launch, pageUrl } from "./cdp.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf("--" + name);
  return i >= 0 ? args[i + 1] : fallback;
};
const has = (name) => args.includes("--" + name);
const [page, out = "shots/out.png"] = args;
const width = Number(flag("w", 1100));
const height = Number(flag("h", 720));
const scale = Number(flag("scale", 1.5));
const hash = [flag("hash", ""), has("stage") ? "view=stage" : "", flag("theme") ? "theme=" + flag("theme") : ""].filter(Boolean).join("&");

const b = await launch({ width, height, scale });
try {
  await b.open(pageUrl(ROOT, page, hash), 400);
  if (flag("eval")) console.log(JSON.stringify(await b.eval(flag("eval"))));
  await b.sleep(Number(flag("wait", 1500)));
  const zoom = flag("zoom");
  if (zoom) {
    const [x, y, w, h] = zoom.split(",").map(Number);
    await b.shot(out, { clip: { x, y, width: w, height: h }, scale: Number(flag("zoom-scale", 3)) });
  } else if (has("stage")) {
    await b.shot(out, { selector: ".iso-plate" });
  } else {
    await b.shot(out);
  }
  console.log("wrote", out);
  if (b.logs.length) console.log("console:", b.logs.slice(0, 8).join("\n  "));
} finally {
  await b.close();
}

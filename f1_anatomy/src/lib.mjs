// Shared page shell, controls and live runtime for every F1 Anatomy figure.
// A figure module builds its geometry with the kit and hands this file a plain description;
// everything around the drawing (navigation, theme, controls, readouts, guided steps) lives here.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../vendor/kit/iso-kit.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/**
 * Turns an ES module into a namespace object inside a page script: imports are dropped (the
 * names they bring in must be supplied by `header`), `export` is stripped, and every exported
 * name is returned. The same geometry and model files therefore run in Node (the build) and in
 * the browser (the live page) without a bundler.
 */
export function nsScript(name, file, header = "") {
  const source = readFileSync(file, "utf8");
  const names = [...source.matchAll(/^export (?:const|let|function|class) ([A-Za-z0-9_$]+)/gm)].map((m) => m[1]);
  const body = source.replace(/^import [^;]*;\s*$/gm, "").replace(/^export /gm, "");
  return `const ${name} = (() => {
${header}
${body}
return { ${names.join(", ")} };
})();`;
}

const KIT_NS = () => [
  nsScript("k", join(ROOT, "vendor/kit/iso-kit.mjs")),
  nsScript("G", join(ROOT, "vendor/kit/lathe.mjs"), "const { cameraOf, iso, pathOf } = k;"),
  nsScript("T", join(ROOT, "vendor/kit/tube.mjs"), "const { iso, pathOf } = k;"),
  nsScript("GL", join(ROOT, "vendor/kit/gl.mjs")),
].join("\n");

export const SITE = "F1 Anatomy";

export const FIGURES = [
  { id: "chassis", num: 1, short: "Chassis", title: "Chassis and Safety" },
  { id: "power-unit", num: 2, short: "Power unit", title: "Power Unit" },
  { id: "aero", num: 3, short: "Aerodynamics", title: "Aerodynamics" },
  { id: "drivetrain", num: 4, short: "Gearbox and wheel", title: "Transmission and Steering" },
  { id: "corner", num: 5, short: "Suspension, brakes, tyres", title: "Suspension, Brakes and Tyres" },
];

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/* ---------- small helpers every figure uses ---------- */

export function tools(P) {
  const S = (paths, style) => k.solidSvg(paths, style);
  const L = (d, style) => k.lineSvg(d, style);
  const D = (points, style) => k.dotsSvg(points, style);
  const at = (point) => k.iso(point, P);
  const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], a[2] ?? 0], P) - k.depthOf([b[0], b[1], b[2] ?? 0], P));
  return { S, L, D, at, byDepth };
}

/** Wraps markup in a named group the page can highlight (`UI.focus("name")`). */
export const group = (name, markup, extra = "") => `<g data-group="${name}"${extra ? " " + extra : ""}>${markup}</g>`;

/* ---------- page shell ---------- */

const CSS = `
*{box-sizing:border-box}
html{color-scheme:dark}
body[data-theme="light"]{color-scheme:light}
body{margin:0;background:var(--anatomy-page);color:var(--anatomy-text);font:14px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.fa-wrap{width:min(1080px,100% - 32px);margin:0 auto;padding:0 0 96px}
.fa-nav{display:flex;flex-wrap:wrap;align-items:center;gap:8px 18px;padding:18px 0;border-bottom:1px solid var(--anatomy-faint)}
.fa-brand{font:600 11px/1 var(--anatomy-mono);letter-spacing:.14em;text-transform:uppercase;color:var(--anatomy-strong);text-decoration:none;display:inline-flex;align-items:center;gap:9px}
.fa-brand i{width:9px;height:9px;background:var(--fa-accent);display:inline-block}
.fa-pills{display:flex;flex-wrap:wrap;gap:4px;flex:1}
.fa-pill{font:11px/1 var(--anatomy-mono);letter-spacing:.04em;color:var(--anatomy-muted);text-decoration:none;padding:7px 10px;border:1px solid transparent}
.fa-pill:hover{color:var(--anatomy-strong)}
.fa-pill[aria-current="page"]{color:var(--anatomy-strong);border-color:var(--anatomy-mid)}
.fa-pill b{color:var(--fa-accent);font-weight:600;margin-right:6px}
.fa-theme{font:11px/1 var(--anatomy-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--anatomy-muted);background:none;border:1px solid var(--anatomy-faint);padding:7px 10px;cursor:pointer}
.fa-theme:hover{color:var(--anatomy-strong);border-color:var(--anatomy-mid)}
.fa-head{padding:34px 0 4px;max-width:760px}
.fa-kicker{font:11px/1 var(--anatomy-mono);letter-spacing:.14em;text-transform:uppercase;color:var(--fa-accent)}
.fa-head h1{margin:12px 0 10px;font:600 30px/1.15 ui-sans-serif,system-ui,sans-serif;letter-spacing:-.01em;color:var(--anatomy-strong)}
.fa-head p{margin:0;font-size:15px;color:var(--anatomy-text)}
.fa-body{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:28px;margin-top:6px}
@media (max-width:820px){.fa-body{grid-template-columns:1fr}}
.fa-panel{border:1px solid var(--anatomy-faint);padding:18px 18px 16px;background:var(--anatomy-card)}
.fa-panel h2{margin:0 0 14px;font:600 11px/1 var(--anatomy-mono);letter-spacing:.14em;text-transform:uppercase;color:var(--anatomy-muted)}
.fa-ctl{display:block;margin:0 0 14px}
.fa-ctl-top{display:flex;justify-content:space-between;gap:12px;font-size:13px;color:var(--anatomy-strong);margin-bottom:6px}
.fa-ctl-top output{font:12px/1.6 var(--anatomy-mono);color:var(--anatomy-text);font-variant-numeric:tabular-nums}
.fa-ctl small{display:block;margin-top:5px;font-size:12px;line-height:1.45;color:var(--anatomy-muted)}
.fa-ctl input[type=range]{width:100%;margin:0;accent-color:var(--fa-accent);background:none;height:18px}
.fa-row{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
.fa-btn,.fa-toggle,.fa-seg button{font:12px/1 var(--anatomy-mono);letter-spacing:.03em;color:var(--anatomy-text);background:var(--anatomy-paper);border:1px solid var(--anatomy-lo);padding:9px 12px;cursor:pointer;transition:border-color .2s,color .2s,background .2s}
.fa-btn:hover,.fa-toggle:hover,.fa-seg button:hover{color:var(--anatomy-strong);border-color:var(--anatomy-mid)}
.fa-btn:active{transform:translateY(1px)}
.fa-btn.primary{border-color:var(--fa-accent);color:var(--anatomy-strong)}
.fa-toggle[aria-checked="true"],.fa-seg button[aria-checked="true"]{color:var(--anatomy-strong);border-color:var(--fa-accent);background:var(--fa-accent-soft)}
.fa-toggle i{display:inline-block;width:7px;height:7px;margin-left:8px;background:var(--anatomy-lo);vertical-align:1px}
.fa-toggle[aria-checked="true"] i{background:var(--fa-accent)}
.fa-seg{display:flex;flex-wrap:wrap;gap:0;margin:0 0 14px}
.fa-seg button{margin-right:-1px}
.fa-seg-label{width:100%;font-size:13px;color:var(--anatomy-strong);margin-bottom:6px}
.fa-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:4px}
.fa-stat{border:1px solid var(--anatomy-faint);padding:10px 11px 9px;background:var(--anatomy-paper)}
.fa-stat span{display:block;font:10.5px/1.2 var(--anatomy-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--anatomy-muted)}
.fa-stat b{display:block;margin-top:6px;font:500 17px/1.1 var(--anatomy-mono);color:var(--anatomy-strong);font-variant-numeric:tabular-nums}
.fa-stat b em{font-style:normal;font-size:11px;color:var(--anatomy-muted);margin-left:4px}
.fa-bar{display:block;height:2px;margin-top:8px;background:var(--anatomy-faint);overflow:hidden}
.fa-bar u{display:block;height:100%;width:0;background:var(--fa-accent);transition:width .12s linear}
.fa-steps{list-style:none;margin:0;padding:0;counter-reset:step}
.fa-steps li{counter-increment:step;margin:0 0 6px}
.fa-step{display:grid;grid-template-columns:22px 1fr;gap:10px;width:100%;text-align:left;padding:10px 10px;background:none;border:1px solid transparent;color:var(--anatomy-text);cursor:pointer;font:inherit}
.fa-step::before{content:counter(step);font:600 11px/1.5 var(--anatomy-mono);color:var(--anatomy-muted)}
.fa-step:hover{border-color:var(--anatomy-faint)}
.fa-step[aria-current="true"]{border-color:var(--anatomy-mid);background:var(--anatomy-paper)}
.fa-step[aria-current="true"]::before{color:var(--fa-accent)}
.fa-step b{display:block;color:var(--anatomy-strong);font-weight:600;font-size:13.5px}
.fa-step span{display:block;margin-top:2px;font-size:13px;line-height:1.5;color:var(--anatomy-text)}
.fa-play{margin-top:10px}
.fa-note{margin:14px 0 0;font-size:12px;line-height:1.5;color:var(--anatomy-muted)}
.fa-stage{touch-action:pan-y;outline:none}
.fa-stage:focus-visible{outline:1px solid var(--anatomy-muted);outline-offset:6px}
.fa-visually-hidden{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.iso{--fa-accent:#e10600;--fa-accent-soft:#e1060014;--fa-compound:#e5484d}
.iso .fa-rubber{--anatomy-paper:#0d0d0e;--anatomy-face:#141416;--anatomy-top:#1a1a1d;--anatomy-shade-0:#0a0a0b;--anatomy-shade-1:#0e0e10;--anatomy-shade-2:#121214;--anatomy-shade-3:#17171a;--anatomy-lit-top:#26262a;--anatomy-lit-shade:#1a1a1d;--anatomy-hi:#6c6c72;--anatomy-mid:#4a4a50;--anatomy-lo:#34343a;--anatomy-faint:#26262b}
.iso[data-theme="light"] .fa-rubber{--anatomy-paper:#3a3a3f;--anatomy-face:#414146;--anatomy-top:#4a4a50;--anatomy-shade-0:#2c2c30;--anatomy-shade-1:#323236;--anatomy-shade-2:#38383d;--anatomy-shade-3:#404045;--anatomy-lit-top:#55555b;--anatomy-lit-shade:#46464b;--anatomy-hi:#9a9aa2;--anatomy-mid:#76767d;--anatomy-lo:#5a5a61;--anatomy-faint:#4c4c52}
.iso[data-theme="light"]{--fa-accent:#d40500;--fa-accent-soft:#d4050010}
.fa-menu-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:16px;margin-top:30px}
.fa-card{display:block;color:inherit;text-decoration:none;border:1px solid var(--anatomy-faint);background:var(--anatomy-card);transition:border-color .2s,transform .3s var(--anatomy-ease)}
.fa-card:hover,.fa-card:focus-visible{border-color:var(--anatomy-mid);transform:translateY(-2px);outline:none}
.fa-card-art{display:block;aspect-ratio:600/330;padding:14px;border-bottom:1px solid var(--anatomy-faint);position:relative}
.fa-card-art img{display:block;width:100%;height:100%;object-fit:contain}
.fa-card-num{position:absolute;top:12px;left:14px;font:600 11px/1 var(--anatomy-mono);color:var(--fa-accent);letter-spacing:.08em}
.fa-card-text{display:block;padding:14px 16px 16px}
.fa-card-text b{display:block;font:600 16px/1.25 ui-sans-serif,system-ui,sans-serif;color:var(--anatomy-strong)}
.fa-card-text span{display:block;margin-top:6px;font-size:13px;line-height:1.5;color:var(--anatomy-text)}
.fa-card-text em{display:block;margin-top:10px;font:11px/1.4 var(--anatomy-mono);font-style:normal;letter-spacing:.04em;color:var(--anatomy-muted)}
.fa-foot{margin-top:40px;font:11px/1.6 var(--anatomy-mono);color:var(--anatomy-muted)}
.iso-figure{margin:22px 0 22px}
[data-group][data-on] .iso-solid>.iso-fill{fill:var(--anatomy-face)}
[data-group][data-on] .iso-solid>.iso-top{fill:var(--anatomy-lit-top)}
[data-group][data-on] .iso-solid>.iso-shade{fill:var(--anatomy-lit-shade)}
[data-group][data-on] .iso-solid>.iso-edge{stroke:var(--fa-accent);stroke-width:1.3px}
.fa-plain{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:0;margin:22px 0 4px;border:1px solid var(--anatomy-faint);background:var(--anatomy-card)}
.fa-plain>div{padding:18px 20px}
.fa-plain>div+div{border-left:1px solid var(--anatomy-faint)}
.fa-plain h2{margin:0 0 10px;font:600 11px/1 var(--anatomy-mono);letter-spacing:.14em;text-transform:uppercase;color:var(--fa-accent)}
.fa-plain p{margin:0 0 9px;font-size:15px;line-height:1.6;color:var(--anatomy-strong)}
.fa-plain p.dim{font-size:13.5px;color:var(--anatomy-text)}
.fa-plain ol{margin:0;padding-left:20px;font-size:14px;line-height:1.55;color:var(--anatomy-text)}
.fa-plain li{margin:0 0 7px}
.fa-plain li b{color:var(--anatomy-strong);font-weight:600}
@media (max-width:820px){.fa-plain{grid-template-columns:1fr}.fa-plain>div+div{border-left:0;border-top:1px solid var(--anatomy-faint)}}
.fa-help{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:28px;margin-top:28px}
@media (max-width:820px){.fa-help{grid-template-columns:1fr}}
.fa-parts{list-style:none;margin:0;padding:0}
.fa-parts li{margin:0 0 4px}
.fa-part{display:block;width:100%;text-align:left;padding:10px 11px;background:none;border:1px solid transparent;color:var(--anatomy-text);cursor:pointer;font:inherit}
.fa-part:hover,.fa-part:focus-visible{border-color:var(--anatomy-faint);outline:none}
.fa-part[aria-pressed="true"]{border-color:var(--fa-accent);background:var(--anatomy-paper)}
.fa-part b{display:block;color:var(--anatomy-strong);font-weight:600;font-size:14px}
.fa-part span{display:block;margin-top:2px;font-size:13px;line-height:1.5}
.fa-gloss{margin:0}
.fa-gloss dt{margin:12px 0 2px;font:600 12.5px/1.3 var(--anatomy-mono);color:var(--anatomy-strong)}
.fa-gloss dt:first-child{margin-top:0}
.fa-gloss dd{margin:0;font-size:13px;line-height:1.55;color:var(--anatomy-text)}
.fa-stat small{display:block;margin-top:5px;font-size:11.5px;line-height:1.35;color:var(--anatomy-muted);text-transform:none;letter-spacing:0;font-family:ui-sans-serif,system-ui,sans-serif}
.fa-step .fa-tip{color:var(--anatomy-muted)}

.view-stage .fa-nav,.view-stage .fa-head,.view-stage .fa-body,.view-stage .iso-legend{display:none}
.view-stage .fa-wrap{width:100%;padding:0}
.view-stage .iso-figure{margin:0}
@media (max-width:560px){.fa-head h1{font-size:25px}}
`;

const THEME_BOOT = `var t=null;try{t=localStorage.getItem("f1anatomy-theme")}catch(e){}if(t!=="light"&&t!=="dark")t=matchMedia("(prefers-color-scheme: light)").matches?"light":"dark";document.body.setAttribute("data-theme",t)`;

const nav = (current) => `<nav class="fa-nav" aria-label="F1 Anatomy"><a class="fa-brand" href="index.html"><i></i>${SITE}</a><div class="fa-pills">${FIGURES.map((f) => `<a class="fa-pill" href="${f.id}.html"${f.id === current ? ' aria-current="page"' : ""}><b>${f.num}</b>${esc(f.short)}</a>`).join("")}</div><button class="fa-theme" type="button" data-theme-toggle>Theme</button></nav>`;

const THEME_JS = `
(() => {
  const body = document.body, button = document.querySelector("[data-theme-toggle]");
  const label = () => { button.textContent = body.getAttribute("data-theme") === "light" ? "Dark mode" : "Light mode"; };
  button.addEventListener("click", () => {
    const next = body.getAttribute("data-theme") === "light" ? "dark" : "light";
    body.setAttribute("data-theme", next);
    try { localStorage.setItem("f1anatomy-theme", next); } catch (e) {}
    label();
  });
  label();
})();
`;

function controlHtml(c) {
  if (c.type === "range") {
    return `<label class="fa-ctl"><span class="fa-ctl-top"><span>${esc(c.label)}</span><output data-out="${c.id}"></output></span><input type="range" data-ctl="${c.id}" min="${c.min}" max="${c.max}" step="${c.step ?? 1}" value="${c.value}" aria-label="${esc(c.label)}">${c.help ? `<small>${esc(c.help)}</small>` : ""}</label>`;
  }
  if (c.type === "toggle") {
    return `<div class="fa-row"><button class="fa-toggle" type="button" role="switch" data-ctl="${c.id}" aria-checked="${c.value ? "true" : "false"}">${esc(c.label)}<i></i></button></div>${c.help ? `<p class="fa-note" style="margin:-8px 0 14px">${esc(c.help)}</p>` : ""}`;
  }
  if (c.type === "choice") {
    return `<div class="fa-seg" role="radiogroup" aria-label="${esc(c.label)}" data-ctl="${c.id}"><span class="fa-seg-label">${esc(c.label)}</span>${c.options.map((o) => `<button type="button" role="radio" data-v="${esc(o.v)}" aria-checked="${o.v === c.value ? "true" : "false"}">${esc(o.label)}</button>`).join("")}</div>${c.help ? `<p class="fa-note" style="margin:-8px 0 14px">${esc(c.help)}</p>` : ""}`;
  }
  if (c.type === "button") {
    return `<div class="fa-row"><button class="fa-btn${c.primary ? " primary" : ""}" type="button" data-act="${c.id}">${esc(c.label)}</button></div>`;
  }
  throw new Error("unknown control type " + c.type);
}

/* ---------- the live runtime (inlined into every figure page) ---------- */

const RUNTIME = `
const UI = (() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const still = matchMedia("(prefers-reduced-motion: reduce)");
  const stage = $(".fa-stage"), svg = stage.querySelector("svg"), readout = $("[data-readout]");
  const state = {}, defs = {}, listeners = [], actions = {}, hold = { until: 0 };
  FA.controls.forEach((c) => { defs[c.id] = c; state[c.id] = c.value; });
  const fmt = (c, v) => (c.fmt ? c.fmt(v) : Number(v).toFixed(c.decimals ?? 0)) + (c.unit ? " " + c.unit : "");
  const FMT = FA.fmt || {};
  FA.controls.forEach((c) => { if (c.fmtName) c.fmt = FMT[c.fmtName]; });
  function paint(id) {
    const c = defs[id], v = state[id];
    if (c.type === "range") { const out = $('[data-out="' + id + '"]'); if (out) out.textContent = fmt(c, v); const el = $('[data-ctl="' + id + '"]'); if (el && Number(el.value) !== v) el.value = v; }
    if (c.type === "toggle") { const el = $('[data-ctl="' + id + '"]'); if (el) el.setAttribute("aria-checked", v ? "true" : "false"); }
    if (c.type === "choice") $$('[data-ctl="' + id + '"] button').forEach((b) => b.setAttribute("aria-checked", b.dataset.v === String(v) ? "true" : "false"));
  }
  const emit = (id, by) => { paint(id); listeners.forEach((fn) => fn(id, state[id], by)); wake(); };
  const set = (id, value, by = "code") => { const c = defs[id]; if (!c) return; state[id] = c.type === "range" ? Math.min(c.max, Math.max(c.min, Number(value))) : c.type === "toggle" ? !!value : String(value); emit(id, by); };
  $$("[data-ctl]").forEach((el) => {
    const id = el.dataset.ctl, c = defs[id];
    if (!c) return;
    if (c.type === "range") el.addEventListener("input", () => { userTouch(); set(id, el.value, "user"); });
    if (c.type === "toggle") el.addEventListener("click", () => { userTouch(); set(id, !state[id], "user"); });
    if (c.type === "choice") $$("button", el).forEach((b) => b.addEventListener("click", () => { userTouch(); set(id, b.dataset.v, "user"); }));
  });
  $$("[data-act]").forEach((el) => el.addEventListener("click", () => { userTouch(); (actions[el.dataset.act] || (() => {}))(); wake(); }));
  FA.controls.forEach((c) => paint(c.id));

  /* stats */
  const stat = (id, text, fraction) => {
    const el = $('[data-stat="' + id + '"]'); if (!el) return;
    if (el.dataset.last !== text) { el.dataset.last = text; el.firstChild.nodeValue = text; }
    const bar = $('[data-bar="' + id + '"]'); if (bar && fraction != null) bar.style.width = Math.max(0, Math.min(1, fraction)) * 100 + "%";
  };
  const say = (text) => { if (readout && readout.textContent !== text) readout.textContent = text; };

  /* highlighting: the groups being talked about get a lit finish with a red outline.
     Three things can ask: the walk-through step, a hovered or picked entry of the parts list, and
     the figure's own code. The last one asked wins; the page keeps the others to come back to. */
  let focused = [];
  const focus = (names) => {
    const list = names ? [].concat(names) : [];
    if (list.join() === focused.join()) return;
    focused = list;
    $$("[data-group]", svg).forEach((g) => (list.includes(g.dataset.group) ? g.setAttribute("data-on", "") : g.removeAttribute("data-on")));
  };
  let stepFocus = null, picked = null, hover = null;
  const applyFocus = () => focus(hover || picked || stepFocus);

  /* loop */
  let frame = 0, last = 0, visible = false, tickFn = null, alive = () => true;
  const perf = { ms: 0, frames: 0 };
  const frameFn = (now) => {
    const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
    last = now;
    const t0 = performance.now();
    const keep = tickFn(dt, still.matches, now / 1000);
    perf.ms = perf.ms * 0.9 + (performance.now() - t0) * 0.1; perf.frames++;
    frame = visible && keep !== false ? requestAnimationFrame(frameFn) : 0;
  };
  function wake() { if (!frame && visible && tickFn) { last = 0; frame = requestAnimationFrame(frameFn); } }
  let touched = false, started = false;
  new IntersectionObserver((entries) => {
    visible = entries[entries.length - 1].isIntersecting;
    if (visible) wake();
    // the walk-through starts by itself the first time the figure is seen, until the reader touches anything
    if (visible && !started && !touched && !still.matches && FA.steps.length && !location.hash.includes("step=") && !location.hash.includes("noauto")) {
      started = true;
      setTimeout(() => { if (!touched) play(); }, 1600);
    }
  }, { rootMargin: "120px 0px" }).observe(stage);
  stage.addEventListener("pointerdown", () => { touched = true; });

  /* guided steps */
  const steps = FA.steps || [];
  let current = -1, playing = false, playTimer = 0;
  const stepButtons = $$(".fa-step");
  function go(i, by) {
    if (!steps.length) return;
    current = (i + steps.length) % steps.length;
    stepButtons.forEach((b, j) => b.setAttribute("aria-current", j === current ? "true" : "false"));
    const s = steps[current];
    Object.entries(s.set || {}).forEach(([id, v]) => set(id, v, "step"));
    stepFocus = s.focus || null;
    applyFocus();
    if (s.run && actions[s.run]) actions[s.run]();
    clearTimers();
    (s.then || []).forEach((t) => timers.push(setTimeout(() => {
      Object.entries(t.set || {}).forEach(([id, v]) => set(id, v, "step"));
      if (t.focus) { stepFocus = t.focus; applyFocus(); }
      if (t.run && actions[t.run]) actions[t.run]();
      wake();
    }, t.after * 1000)));
    listeners.forEach((fn) => fn("step", current, by || "step"));
    wake();
  }
  const timers = [];
  function clearTimers() { while (timers.length) clearTimeout(timers.pop()); }
  stepButtons.forEach((b, i) => b.addEventListener("click", () => { stop(); go(i, "user"); }));
  const playBtn = $("[data-play]");
  function stop() { playing = false; clearTimeout(playTimer); clearTimers(); if (playBtn) { playBtn.textContent = "Play the walk-through"; playBtn.setAttribute("aria-pressed", "false"); } }
  function play() {
    playing = true; if (playBtn) { playBtn.textContent = "Stop"; playBtn.setAttribute("aria-pressed", "true"); }
    const next = () => { if (!playing) return; go(current + 1, "tour"); playTimer = setTimeout(next, steps[current].dwell ? steps[current].dwell * 1000 : 6500); };
    next();
  }
  if (playBtn) playBtn.addEventListener("click", () => (playing ? stop() : play()));
  function userTouch() { touched = true; if (playing) stop(); clearTimers(); hold.until = performance.now() + 2500; }

  /* the parts list: hover to find a part, click to keep it picked and read about it */
  const partButtons = $$(".fa-part");
  const showPart = (button) => {
    partButtons.forEach((b) => b.setAttribute("aria-pressed", b === button ? "true" : "false"));
  };
  partButtons.forEach((button) => {
    const groups = button.dataset.groups.split(" ");
    button.addEventListener("pointerenter", () => { hover = groups; applyFocus(); });
    button.addEventListener("pointerleave", () => { hover = null; applyFocus(); });
    button.addEventListener("focus", () => { hover = groups; applyFocus(); });
    button.addEventListener("blur", () => { hover = null; applyFocus(); });
    button.addEventListener("click", () => {
      touched = true; stop(); clearTimers();
      const same = picked && picked.join() === groups.join();
      picked = same ? null : groups;
      showPart(same ? null : button);
      applyFocus();
    });
  });
  /* clicking a part of the drawing picks the matching entry of the list */
  svg.addEventListener("click", (event) => {
    const g = event.target.closest && event.target.closest("[data-group]");
    if (!g) return;
    const button = partButtons.find((b) => b.dataset.groups.split(" ").includes(g.dataset.group));
    if (!button) return;
    touched = true; stop(); clearTimers();
    picked = button.dataset.groups.split(" ");
    showPart(button);
    applyFocus();
    button.scrollIntoView({ block: "nearest", behavior: still.matches ? "auto" : "smooth" });
  });

  const api = {
    $, $$, stage, svg, state, set, stat, say, focus, still, hold, perf,
    on: (fn) => listeners.push(fn),
    action: (id, fn) => { actions[id] = fn; },
    loop: (fn) => { tickFn = fn; wake(); },
    wake, go, play, stop,
    get step() { return current; },
    get playing() { return playing; },
  };
  /* debug and share hooks: #speed=300&brake=100&step=2&theme=light&view=stage */
  try {
    const q = new URLSearchParams(location.hash.slice(1));
    if (q.get("theme")) document.body.setAttribute("data-theme", q.get("theme"));
    if (q.get("view") === "stage") document.documentElement.classList.add("view-stage");
    q.forEach((value, key) => { if (defs[key]) set(key, defs[key].type === "toggle" ? value === "1" || value === "true" : value, "hash"); });
    if (q.get("step") != null) setTimeout(() => go(Number(q.get("step")), "hash"), 0);
    if (q.get("do")) setTimeout(() => (actions[q.get("do")] || (() => {}))(), 0);
  } catch (e) {}
  window.__faUI = api;
  return api;
})();
`;

export const MATH = `
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => t * t * (3 - 2 * t);
const follow = (cur, target, dt, seconds, calm) => (calm ? target : cur + (target - cur) * (1 - Math.exp(-dt / seconds)));
// critically damped spring on { x, v }, same as the kit's settle but never snaps early
const spring = (s, target, dt, stiffness, damping, calm) => {
  if (calm) { s.x = target; s.v = 0; return false; }
  for (let i = 0; i < 4; i++) { s.v += ((target - s.x) * stiffness - s.v * damping) * dt / 4; s.x += s.v * dt / 4; }
  if (Math.abs(target - s.x) < 5e-4 && Math.abs(s.v) < 5e-4) { s.x = target; s.v = 0; return false; }
  return true;
};
`;

/**
 * figure: {
 *   id, title, kicker, intro, fig, plateTitle, hint, readout,
 *   width, height, svg (inner svg body), label,
 *   keys: [{mark,label}], caption,
 *   controls: [...], stats: [{id,label,unit}], steps: [{title,text,set,focus,dwell}], note,
 *   data: object (JSON, exposed to live code as `DATA`), live: string (JS source)
 * }
 */
export function figurePage(f0, { extraScript = "" } = {}) {
  const ex = f0.explain || {};
  const f = {
    ...f0,
    controls: (f0.controls || []).map((c) => ({ ...c, help: (ex.controls && ex.controls[c.id]) || c.help })),
    steps: (f0.steps || []).map((s, i) => ({ ...s, ...((ex.steps && ex.steps[i]) || {}) })),
    stats: (f0.stats || []).map((s) => ({ ...s, hint: ex.stats && ex.stats[s.id], label: (ex.statLabels && ex.statLabels[s.id]) || s.label })),
  };
  return buildFigurePage(f, ex, { extraScript });
}

function buildFigurePage(f, ex, { extraScript = "" } = {}) {
  const svg = k.figureSvg({ width: f.width, height: f.height, label: f.label, body: f.svg });
  const plate = k.plateHtml({
    fig: f.fig,
    title: f.plateTitle,
    hint: f.hint,
    readout: f.readout,
    keys: f.keys,
    caption: f.caption,
    body: `<div class="fa-stage iso-stage" tabindex="0" role="group" aria-label="${esc(f.stageLabel || f.title)}">${svg}</div>`,
  });
  const controls = f.controls || [];
  const stats = f.stats || [];
  const steps = f.steps || [];
  const controlsPanel = `<section class="fa-panel" aria-label="Controls"><h2>Try it</h2>${controls.map(controlHtml).join("")}${stats.length ? `<div class="fa-stats">${stats.map((s) => `<div class="fa-stat"${s.hint ? ` title="${esc(s.hint)}"` : ""}><span>${esc(s.label)}</span><b data-stat="${s.id}">–<em>${esc(s.unit || "")}</em></b>${s.bar ? `<i class="fa-bar"><u data-bar="${s.id}"></u></i>` : ""}${s.hint ? `<small>${esc(s.hint)}</small>` : ""}</div>`).join("")}</div>` : ""}${f.note ? `<p class="fa-note">${esc(f.note)}</p>` : ""}</section>`;
  const stepsPanel = `<section class="fa-panel" aria-label="Walk-through"><h2>How it works</h2><ol class="fa-steps">${steps.map((s, i) => `<li><button class="fa-step" type="button" aria-current="false"><div><b>${esc(s.title)}</b><span>${esc(s.text)}</span></div></button></li>`).join("")}</ol>${steps.length ? `<button class="fa-btn fa-play" type="button" data-play aria-pressed="false">Play the walk-through</button>` : ""}</section>`;
  const plain = ex.plain || null;
  const plainBox = plain
    ? `<section class="fa-plain" aria-label="In plain English"><div><h2>In plain English</h2>${(plain.idea || []).map((t, i) => `<p${i ? ' class="dim"' : ""}>${esc(t)}</p>`).join("")}</div><div><h2>Try this first</h2><ol>${(plain.tryFirst || []).map((t) => `<li>${t.split("|").length > 1 ? `<b>${esc(t.split("|")[0])}</b> ${esc(t.split("|")[1])}` : esc(t)}</li>`).join("")}</ol></div></section>`
    : "";
  const parts = ex.parts || [];
  const glossary = ex.glossary || [];
  const helpPanels =
    parts.length || glossary.length
      ? `<div class="fa-help">${parts.length ? `<section class="fa-panel" aria-label="What you are looking at"><h2>What you are looking at</h2><p class="fa-note" style="margin:-4px 0 12px">Hover a name to find it in the drawing. Click it, or click the part itself, to keep it marked.</p><ul class="fa-parts">${parts.map((p) => `<li><button class="fa-part" type="button" aria-pressed="false" data-groups="${esc(p.groups.join(" "))}"><b>${esc(p.name)}</b><span>${esc(p.text)}</span></button></li>`).join("")}</ul></section>` : ""}${glossary.length ? `<section class="fa-panel" aria-label="Words used here"><h2>Words used here</h2><dl class="fa-gloss">${glossary.map((g) => `<dt>${esc(g.term)}</dt><dd>${esc(g.text)}</dd>`).join("")}</dl></section>` : ""}</div>`
      : "";
  const body = `<div class="fa-wrap">${nav(f.id)}<header class="fa-head"><div class="fa-kicker">${esc(f.fig)} · ${esc(f.kicker)}</div><h1>${esc(f.title)}</h1>${plainBox ? "" : `<p>${esc(f.intro)}</p>`}</header>${plainBox}${plate}<div class="fa-body">${controlsPanel}${stepsPanel}</div>${helpPanels}</div>`;
  const fa = {
    controls: controls.map((c) => ({ ...c })),
    steps: steps.map((s) => ({ set: s.set, focus: s.focus, run: s.run, dwell: s.dwell, then: s.then })),
  };
  const script = [
    KIT_NS(),
    ...(f.modules || []).map((m) => nsScript(m.name, m.file, m.header || "")),
    `const FA = ${JSON.stringify(fa)};`,
    `const DATA = ${JSON.stringify(f.data ?? {})};`,
    MATH,
    RUNTIME,
    THEME_JS,
    extraScript,
    `{\n${f.live}\n}`,
  ].join("\n");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(f.title)} · ${SITE}</title><style>${k.ISO_CSS}${CSS}</style></head><body class="iso" data-theme="dark"><script>${THEME_BOOT}</script>${body}<script type="module">${script.replace(/<\/(script)/gi, "<\\/$1")}</script></body></html>`;
}

/* ---------- menu page ---------- */

export function menuPage(cards, overview) {
  const body = `<div class="fa-wrap">${nav("")}<header class="fa-head"><div class="fa-kicker">Interactive explainer</div><h1>${esc(overview.title)}</h1><p>${esc(overview.lead)}</p></header><section class="fa-plain" aria-label="How the five parts fit together"><div><h2>How the five parts fit together</h2><ol>${overview.how.map((h) => `<li><b>${esc(h.name)}</b> ${esc(h.text)}</li>`).join("")}</ol></div><div><h2>How to use these pages</h2><p class="dim">Each page has a drawing you can poke, sliders underneath, and a short guided tour that starts by itself.</p><p class="dim">Under the drawing, the list of parts tells you what each thing is. Hover a name to see where it is. Any word you do not know is in the short word list at the bottom.</p></div></section><div class="fa-menu-grid">${cards
    .map(
      (c) =>
        `<a class="fa-card" href="${c.id}.html"><span class="fa-card-art"><span class="fa-card-num">0${c.num}</span><img data-thumb="${c.id}" src="thumbs/${c.id}-dark.svg" alt="" loading="lazy"></span><span class="fa-card-text"><b>${esc(c.title)}</b><span>${esc(c.blurb)}</span><em>${esc(c.tryit)}</em></span></a>`,
    )
    .join("")}</div><p class="fa-foot">Drawings are isometric and built with the Anatomy kit. Quantities are rounded, typical 2020s Formula 1 figures; regulations differ by season.</p></div>`;
  const swap = `const sync=()=>{const t=document.body.getAttribute("data-theme")==="light"?"light":"dark";document.querySelectorAll("[data-thumb]").forEach((i)=>{i.src="thumbs/"+i.dataset.thumb+"-"+t+".svg"})};new MutationObserver(sync).observe(document.body,{attributes:true,attributeFilter:["data-theme"]});sync();`;
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${SITE}</title><style>${k.ISO_CSS}${CSS}</style></head><body class="iso" data-theme="dark"><script>${THEME_BOOT}</script>${body}<script>${THEME_JS}${swap}</script></body></html>`;
}

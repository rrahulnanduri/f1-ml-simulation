// Live behaviour of the chassis figure. MODEL, GEO, UI, FA, DATA, k, G, T are in scope.
const P = DATA.P;
const items = GEO.items(P);
const nodes = {};
UI.$$("[data-item]").forEach((el) => (nodes[el.dataset.item] = el));

const U = GEO.U;
// carbon fibre is nearly black: its own token set for the shell, like the tyre in the corner figure
const carbon = document.createElement("style");
carbon.textContent = `.iso .fa-carbon{--anatomy-paper:#222226;--anatomy-face:#28282c;--anatomy-top:#303035;--anatomy-shade-0:#1a1a1e;--anatomy-shade-1:#1e1e22;--anatomy-shade-2:#232327;--anatomy-shade-3:#29292d;--anatomy-hi:#8c8c94;--anatomy-mid:#6a6a72;--anatomy-lo:#4c4c54;--anatomy-faint:#38383e}
.iso[data-theme="light"] .fa-carbon{--anatomy-paper:#3c3c41;--anatomy-face:#424247;--anatomy-top:#4c4c52;--anatomy-shade-0:#2e2e32;--anatomy-shade-1:#343438;--anatomy-shade-2:#3a3a3f;--anatomy-shade-3:#424247;--anatomy-hi:#a2a2aa;--anatomy-mid:#7c7c84;--anatomy-lo:#5e5e66;--anatomy-faint:#505057}`;
document.head.appendChild(carbon);
const SLOW = 0.07; // the 60 ms crash plays at 7% speed
const SIDE_FACE = 90;
const sim = { phase: "idle", t: 0, kind: "frontal", res: null, simT: 0, carDx: 0, crushU: 0, lean: 0, ramY: GEO.RAM_REST, panelC: 0, intrU: 0, haloDefl: 0, engaged: 0, cursor: 0, skin: 0.22, plot: null, plotKey: "" };
const lastKeys = {};

const params = () => {
  const s = UI.state;
  return { test: s.test, v: s.speed, force: s.nose * 1000, load: s.load * 1000 };
};

function predict() {
  const p = params();
  if (p.test === "frontal") return MODEL.frontal(p.v, p.force);
  if (p.test === "side") return MODEL.side(Math.min(p.v, MODEL.TEST.sideSpeedMax));
  return MODEL.halo(p.load);
}

function plotOf(res, test) {
  if (test === "halo" || !res.trace || !res.trace.length) return null;
  return { trace: res.trace, at: (t) => MODEL.traceAt(res.trace, t) };
}

function setIdle() {
  const res = predict();
  const test = UI.state.test;
  sim.phase = "idle";
  sim.kind = test;
  sim.res = res;
  sim.carDx = -(test === "frontal" ? 0 : 0);
  Object.assign(sim, { simT: 0, carDx: 0, crushU: 0, lean: 0, ramY: GEO.RAM_REST, panelC: 0, intrU: 0, haloDefl: 0, engaged: 0, cursor: 0 });
  sim.plot = plotOf(res, test);
  sim.plotKey = test + UI.state.speed + UI.state.nose;
}

function startRun() {
  setIdle();
  sim.phase = "approach";
  sim.t = 0;
  if (sim.kind === "frontal") sim.carDx = -GEO.APPROACH;
  UI.wake();
}
UI.action("run", startRun);

const ease = (x) => x * x * (3 - 2 * x);

function advance(dt) {
  const res = sim.res;
  if (sim.phase === "approach") {
    sim.t += dt;
    if (sim.kind === "frontal") {
      const f = Math.min(1, sim.t / 0.9);
      sim.carDx = -GEO.APPROACH * (1 - f);
      if (f >= 1) { sim.phase = "impact"; sim.simT = 0; }
    } else if (sim.kind === "side") {
      const f = Math.min(1, sim.t / 0.7);
      sim.ramY = GEO.RAM_REST - (GEO.RAM_REST - SIDE_FACE) * ease(f);
      if (f >= 1) { sim.phase = "impact"; sim.simT = 0; }
    } else {
      sim.engaged = ease(Math.min(1, sim.t / 0.9));
      if (sim.t >= 0.9) { sim.phase = "impact"; sim.simT = 0; }
    }
    return;
  }
  if (sim.phase === "impact") {
    if (sim.kind === "halo") {
      sim.simT += dt;
      const f = ease(Math.min(1, sim.simT / 1.8));
      sim.haloDefl = res.defl / U * 8 * f; // drawn at eight times the real deflection
      sim.cursor = f;
      if (sim.simT >= 1.8) sim.phase = "done";
      return;
    }
    sim.simT += dt * SLOW;
    const c = MODEL.traceAt(res.trace, Math.min(sim.simT, res.duration));
    sim.cursor = Math.min(sim.simT, res.duration);
    if (sim.kind === "frontal") {
      sim.carDx = c.x / U;
      sim.crushU = Math.min(c.x, MODEL.TEST.noseCrushable) / U;
      sim.intrU = Math.max(0, c.x - MODEL.TEST.noseCrushable) / U;
      sim.lean = Math.min(0.55, c.r / 0.5);
    } else {
      sim.ramY = SIDE_FACE - c.x / U;
      sim.panelC = Math.min(c.x, MODEL.TEST.sideCrushable) / U;
      sim.intrU = Math.max(0, c.x - MODEL.TEST.sideCrushable) / U;
    }
    if (sim.simT >= res.duration) { sim.phase = "done"; sim.t = 0; }
    return;
  }
  if (sim.phase === "done") {
    sim.t += dt;
    if (sim.kind === "frontal") sim.lean = Math.max(0, sim.lean - dt * 0.4); // the driver settles back into the seat
  }
}

function pose() {
  return GEO.poseOf({ carDx: sim.carDx, crushU: sim.crushU, lean: sim.lean, skin: sim.skin, haloDefl: sim.haloDefl, engaged: sim.engaged, ramY: sim.ramY, panelC: sim.panelC, intrU: sim.intrU, cursor: sim.cursor, plotKey: sim.plotKey, plot: sim.plot });
}

function paint(p) {
  items.forEach((item) => {
    const el = nodes[item.id];
    if (!el) return;
    if (item.xform) {
      const shift = GEO.shiftOf(item, p);
      const key = shift.map((v) => v.toFixed(2)).join();
      if (lastKeys["x" + item.id] !== key) { lastKeys["x" + item.id] = key; el.setAttribute("transform", k.translateAlong(P, shift)); }
    }
    if (item.skin) {
      const o = (0.05 + 0.95 * p.skin).toFixed(3);
      if (lastKeys["o" + item.id] !== o) { lastKeys["o" + item.id] = o; el.style.opacity = o; }
    }
    if (item.live) {
      const key = (item.deps || []).map((d) => (typeof p[d] === "number" ? p[d].toFixed(item.fine || 2) : p[d])).join("|");
      if (lastKeys[item.id] !== key) { lastKeys[item.id] = key; el.innerHTML = item.live(p); }
    }
  });
}

const f1 = (v) => v.toFixed(1);
function stats() {
  const res = sim.res || predict();
  const running = sim.phase === "impact" || sim.phase === "done";
  const test = sim.kind;
  const show = (id, text, frac) => UI.stat(id, text, frac);
  if (test === "halo") {
    const f = sim.phase === "idle" || sim.phase === "approach" ? 0 : sim.cursor;
    const load = (res.load * (running ? f : 1)) / 1000;
    const defl = res.defl * 1000 * (running ? f : 1);
    ["peak", "avg", "crush", "energy", "intrusion", "head"].forEach((id) => show(id, "–", 0));
    show("halo", Math.round(load) + " · " + f1(defl), load / 125);
    show("result", res.pass ? "PASS" : "FAIL", 1);
    UI.say("Halo " + Math.round(load) + " kN · " + f1(defl) + " mm · " + (sim.phase === "idle" ? "ready" : sim.phase === "done" ? "held" : "loading"));
    return;
  }
  const tr = MODEL.traceAt(res.trace, sim.cursor);
  const prog = running ? (sim.phase === "done" ? 1 : Math.min(1, sim.cursor / res.duration)) : 0;
  const view = running ? tr : { g: 0, x: 0, r: 0, v: res.v };
  const peakSoFar = running ? Math.max(0, ...res.trace.filter((p) => p.t <= sim.cursor).map((p) => p.g)) : 0;
  const shown = running ? 1 : 0;
  const peak = running ? peakSoFar : res.peak;
  show("peak", peak.toFixed(0), peak / 80);
  show("avg", res.avg.toFixed(0), res.avg / 40);
  const crush = running ? Math.min(view.x, test === "frontal" ? MODEL.TEST.noseCrushable : MODEL.TEST.sideCrushable) * 1000 : res.crush * 1000;
  show("crush", crush.toFixed(0), crush / 600);
  const e = running ? 0.5 * MODEL.TEST.mass * (res.v * res.v - view.v * view.v) / 1000 : res.energy / 1000;
  show("energy", e.toFixed(0), e / 100);
  const intr = running ? Math.max(0, view.x - (test === "frontal" ? MODEL.TEST.noseCrushable : MODEL.TEST.sideCrushable)) * 1000 : res.intrusion * 1000;
  show("intrusion", intr.toFixed(0), intr / 60);
  const head = running ? view.r * 1000 : res.head * 1000;
  show("head", test === "frontal" ? head.toFixed(0) : "–", head / 150);
  show("halo", "–", 0);
  show("result", res.pass ? "PASS" : "FAIL", 1);
  void shown; void prog;
  const label = test === "frontal" ? "frontal" : "side";
  UI.say(res.v.toFixed(1) + " m/s " + label + " · " + (sim.phase === "idle" ? "ready · predicts " + (res.pass ? "pass" : "fail") : sim.phase === "done" ? (res.pass ? "pass" : "fail") : "running"));
}

let described = "";
function describe() {
  const text = "Crash test rig. " + UI.state.test + " test. " + UI.$("[data-readout]").textContent;
  if (text !== described) { described = text; UI.stage.setAttribute("aria-label", text); }
}

let ticks = 0;
function tick(dt, calm) {
  const target = UI.state.skin / 100;
  sim.skin = calm ? target : follow(sim.skin, target, dt, 0.3, false);
  if (Math.abs(sim.skin - target) < 0.002) sim.skin = target;
  const calmRun = calm && (sim.phase === "approach" || sim.phase === "impact");
  advance(calmRun ? 10 : dt);
  const p = pose();
  paint(p);
  if (ticks++ % 3 === 0 || sim.phase === "done") { stats(); describe(); }
  const busy = sim.phase === "approach" || sim.phase === "impact" || (sim.phase === "done" && sim.t < 0.9) || sim.skin !== target;
  return busy || UI.playing;
}

// a change to a control puts the rig back to its start, so what is drawn always matches what is set
UI.on((id, value, by) => {
  if (id === "step" || !(id in UI.state)) return;
  if (id === "skin") return;
  setIdle();
  paint(pose());
  stats();
  UI.wake();
});

// dragging up and down on the figure sets the speed, or the load for the Halo test
let drag = null;
UI.stage.addEventListener("pointerdown", (e) => {
  const id = UI.state.test === "halo" ? "load" : "speed";
  drag = { y: e.clientY, id, start: UI.state[id] };
  UI.stage.setPointerCapture(e.pointerId);
  UI.stop();
});
UI.stage.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const box = UI.svg.getBoundingClientRect();
  const c = FA.controls.find((x) => x.id === drag.id);
  const span = c.max - c.min;
  UI.set(drag.id, drag.start + ((drag.y - e.clientY) / box.height) * span * 1.6, "user");
});
const release = () => { drag = null; };
UI.stage.addEventListener("pointerup", release);
UI.stage.addEventListener("pointercancel", release);
UI.stage.addEventListener("keydown", (e) => {
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); UI.stop(); startRun(); }
});

setIdle();
sim.skin = UI.state.skin / 100;
UI.loop(tick);
paint(pose());
stats();

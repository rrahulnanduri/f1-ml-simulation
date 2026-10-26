// Live behaviour of the aerodynamics figure. MODEL, GEO, UI, FA, DATA, k, G are in scope.
const P = DATA.P;
const items = GEO.items(P);
const nodes = {};
UI.$$("[data-item]").forEach((el) => (nodes[el.dataset.item] = el));

const s0 = UI.state;
const sim = { v: s0.speed / 3.6, ride: s0.ride, fw: s0.fw, rw: s0.rw, drs: s0.drs ? 1 : 0, xray: s0.xray ? 1 : 0, belt: 0, spin: 0, phase: 0 };
const lastKeys = {};
let ticks = 0;
let forces = null;

function pose() {
  const f = forces || MODEL.forces({ ride: sim.ride, fw: sim.fw, rw: sim.rw, drs: sim.drs }, sim.v);
  return GEO.poseOf({ ride: sim.ride, fw: sim.fw, rw: sim.rw, drs: sim.drs, xray: sim.xray, spin: sim.spin, belt: sim.belt, f, dial: sim.ride });
}

function paint(p) {
  items.forEach((item) => {
    const el = nodes[item.id];
    if (!el) return;
    if (item.xform) {
      const key = p.carShift.toFixed(2);
      if (lastKeys["x" + item.id] !== key) {
        lastKeys["x" + item.id] = key;
        el.setAttribute("transform", k.translateAlong(P, GEO.shiftOf(item, p)));
      }
    }
    if (item.cls === "fa-floor" || item.cls === "fa-xray") {
      const o = (item.cls === "fa-floor" ? 1 - 0.86 * p.xray : p.xray).toFixed(3);
      if (lastKeys["o" + item.id] !== o) { lastKeys["o" + item.id] = o; el.style.opacity = o; }
    }
    if (item.live) {
      const key = (item.deps || []).map((d) => (typeof p[d] === "number" ? p[d].toFixed(item.fine || 2) : p[d])).join("|");
      if (lastKeys[item.id] !== key) {
        lastKeys[item.id] = key;
        el.innerHTML = item.live(p);
      }
    }
  });
}

function stats(f) {
  const kN = (n) => (n / 1000).toFixed(1);
  UI.stat("down", kN(f.down), f.down / 20000);
  UI.stat("split", Math.round((f.front / f.down || 0) * 100) + " / " + Math.round((f.rear / f.down || 0) * 100));
  UI.stat("drag", kN(f.drag), f.drag / 6000);
  UI.stat("ld", f.ld.toFixed(1));
  UI.stat("floor", Math.round(f.floorShare * 100) + "", f.floorShare);
  const open = MODEL.toKmh(MODEL.topSpeed({ ride: sim.ride, fw: sim.fw, rw: sim.rw, drs: 1 }));
  const shut = MODEL.toKmh(MODEL.topSpeed({ ride: sim.ride, fw: sim.fw, rw: sim.rw, drs: 0 }));
  UI.stat("vmax", Math.round(MODEL.toKmh(MODEL.topSpeed({ ride: sim.ride, fw: sim.fw, rw: sim.rw, drs: sim.drs }))) + "");
  UI.stat("gain", "+" + Math.round(open - shut));
  UI.say(Math.round(sim.v * 3.6) + " km/h · down " + kN(f.down) + " kN · drag " + kN(f.drag) + " kN" + (MODEL.porpoising(sim.ride) ? " · porpoising" : ""));
}

let described = "";
function describe(f) {
  const text = "Wind tunnel at " + Math.round(sim.v * 3.6) + " kilometres per hour: downforce " + (f.down / 1000).toFixed(1) + " kilonewtons, drag " + (f.drag / 1000).toFixed(1) + ", DRS " + (UI.state.drs ? "open" : "closed") + ".";
  if (text !== described) { described = text; UI.stage.setAttribute("aria-label", text); }
}

let flowNodes = [];
function collectFlow() { flowNodes = UI.$$(".fa-flow", UI.svg); }

function tick(dt, calm, now) {
  const s = UI.state;
  const target = s.speed / 3.6;
  const wobble = MODEL.porpoising(sim.ride) && s.speed > 120 && !calm ? 1.4 * Math.sin(now * 14) * Math.min(1, s.speed / 250) : 0; // mm
  sim.v = follow(sim.v, target, dt, 0.5, calm);
  if (Math.abs(sim.v - target) < 0.02) sim.v = target;
  sim.ride = follow(sim.ride, s.ride, dt, 0.22, calm);
  sim.fw = follow(sim.fw, s.fw, dt, 0.3, calm);
  sim.rw = follow(sim.rw, s.rw, dt, 0.3, calm);
  sim.drs = follow(sim.drs, s.drs ? 1 : 0, dt, 0.3, calm);
  if (Math.abs(sim.drs - (s.drs ? 1 : 0)) < 0.002) sim.drs = s.drs ? 1 : 0;
  sim.xray = follow(sim.xray, s.xray ? 1 : 0, dt, 0.35, calm);
  const shown = sim.ride + wobble;
  forces = MODEL.forces({ ride: shown, fw: sim.fw, rw: sim.rw, drs: sim.drs }, sim.v);
  const omega = sim.v / 0.33;
  sim.spin += (omega > 0 ? 0.4 + 6.8 * Math.pow(omega / 252, 0.6) : 0) * dt;
  sim.belt += (calm ? 0 : sim.v * 0.55) * dt;
  sim.phase += (calm ? 0 : 6 + sim.v * 0.75) * dt;
  const p = pose();
  p.carShift = shown / 10 - GEO.H0;
  paint(p);
  if (!flowNodes.length || !flowNodes[0].isConnected) collectFlow();
  const off = (-sim.phase).toFixed(1);
  const fast = (-sim.phase * 2).toFixed(1);
  flowNodes.forEach((n) => { n.style.strokeDashoffset = n.classList.contains("fa-fast") ? fast : off; });
  if (ticks++ % 3 === 0) { stats(forces); describe(forces); }
  const calmDown = sim.v === target && Math.abs(sim.ride - s.ride) < 0.01 && Math.abs(sim.fw - s.fw) < 0.01 && Math.abs(sim.rw - s.rw) < 0.01 && sim.drs === (s.drs ? 1 : 0) && Math.abs(sim.xray - (s.xray ? 1 : 0)) < 0.002 && !MODEL.porpoising(sim.ride);
  return !(calmDown && target < 0.5) || UI.playing;
}

/* direct manipulation: click the rear wing for DRS, drag up and down for the ride height */
const hit = GEO.hitAreas(P);
let drag = null;
const toBox = (e) => { const r = UI.svg.getBoundingClientRect(); const sc = DATA.width / r.width; return [(e.clientX - r.left) * sc, (e.clientY - r.top) * sc]; };
UI.stage.addEventListener("pointerdown", (e) => {
  const [x, y] = toBox(e);
  drag = { y: e.clientY, ride: UI.state.ride, moved: false, wing: Math.hypot(x - hit.rearWing[0], y - hit.rearWing[1]) < 56, id: e.pointerId };
  UI.stage.setPointerCapture(e.pointerId);
});
UI.stage.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dy = e.clientY - drag.y;
  if (Math.abs(dy) > 5) drag.moved = true;
  if (drag.moved) { UI.stop(); UI.set("ride", drag.ride - dy * 0.22, "user"); }
});
const release = () => {
  if (drag && !drag.moved && drag.wing) { UI.stop(); UI.set("drs", !UI.state.drs, "user"); }
  drag = null;
};
UI.stage.addEventListener("pointerup", release);
UI.stage.addEventListener("pointercancel", () => { drag = null; });
UI.stage.addEventListener("keydown", (e) => {
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); UI.stop(); UI.set("drs", !UI.state.drs, "user"); }
  if (e.key === "ArrowUp") { e.preventDefault(); UI.set("ride", UI.state.ride + 2, "user"); }
  if (e.key === "ArrowDown") { e.preventDefault(); UI.set("ride", UI.state.ride - 2, "user"); }
  if (e.key === "ArrowRight") { e.preventDefault(); UI.set("speed", UI.state.speed + 10, "user"); }
  if (e.key === "ArrowLeft") { e.preventDefault(); UI.set("speed", UI.state.speed - 10, "user"); }
});

UI.loop(tick);
UI.on(() => UI.wake());
forces = MODEL.forces({ ride: sim.ride, fw: sim.fw, rw: sim.rw, drs: sim.drs }, sim.v);
paint(pose());
collectFlow();
stats(forces);

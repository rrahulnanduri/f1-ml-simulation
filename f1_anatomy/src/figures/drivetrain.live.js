// Live behaviour of the gearbox figure. M (model), GEO, UI, FA, DATA, k, G, T are in scope.
const P = DATA.P;
const items = GEO.items(P);
const nodes = {};
UI.$$("[data-item]").forEach((el) => (nodes[el.dataset.item] = el));

const car = { v: 0, gear: 1, rpm: 1500, limiter: 0, shiftAt: -10, traction: 0, accel: 0 };
const sim = { phase: 0, shaft: 0, drum: 1, rpm: 1500, detach: 0, padU: 0, padD: 0, now: 0 };
const lastKeys = {};
const MODES = M.MODES.map((m) => m.id);
const DIFFS = M.DIFF.map((d) => d.id);
const detached = () => UI.state.detach;

function pose() {
  const s = UI.state;
  const rings = [0, 1, 2, 3].map((kk) => M.trackAt(kk, sim.drum));
  return GEO.poseOf({
    gear: car.gear,
    rpm: sim.rpm,
    speed: car.v * 3.6,
    phase: sim.phase,
    shaftPhase: sim.shaft,
    drum: sim.drum,
    rings,
    detach: sim.detach,
    padU: sim.padU,
    padD: sim.padD,
    bb: s.bb - 54,
    diff: DIFFS.indexOf(s.diff),
    mode: MODES.indexOf(s.mode),
    limiter: car.limiter,
    leds: Math.max(0, Math.min(15, Math.round(((sim.rpm - 9000) / (15000 - 9000)) * 15))),
  });
}

function paint(p) {
  items.forEach((item) => {
    const el = nodes[item.id];
    if (!el) return;
    if (item.xform) {
      const shift = GEO.shiftOf(item, p);
      const key = shift.map((v) => v.toFixed(2)).join();
      if (lastKeys["x" + item.id] !== key) {
        lastKeys["x" + item.id] = key;
        el.setAttribute("transform", k.translateAlong(P, shift));
      }
    }
    if (item.live) {
      const key = (item.deps || []).map((d) => (typeof p[d] === "number" ? p[d].toFixed(item.fine ?? 2) : p[d])).join("|");
      if (lastKeys[item.id] !== key) {
        lastKeys[item.id] = key;
        el.innerHTML = item.live(p);
      }
    }
  });
  // the layshaft gear that is locked to its shaft takes the lit finish
  for (let i = 1; i <= 8; i++) {
    const on = i === car.gear && sim.detach < 0.5;
    const el = nodes["lay-gear-" + i];
    if (el && lastKeys["lit" + i] !== on) {
      lastKeys["lit" + i] = on;
      el.querySelectorAll(".iso-solid").forEach((s) => (on ? s.setAttribute("data-lit", "") : s.removeAttribute("data-lit")));
    }
  }
}

const f1 = (n, d = 0) => Number(n).toFixed(d);
function stats() {
  const s = UI.state;
  const mode = M.MODES[MODES.indexOf(s.mode)];
  const torque = M.crankTorque(car.gear, Math.max(sim.rpm, 3000), s.throttle / 100, mode);
  const power = (torque * M.omega(sim.rpm)) / 1000;
  const decel = car.accel < 0 ? -car.accel : 0;
  UI.stat("speed", f1(car.v * 3.6), car.v / (350 / 3.6));
  UI.stat("rpm", f1(sim.rpm), sim.rpm / M.REV_LIMIT);
  UI.stat("gear", detached() ? "—" : String(car.gear) + " / 8", car.gear / 8);
  UI.stat("ratio", f1(M.RATIOS[car.gear - 1], 2) + " : 1");
  UI.stat("torque", f1(torque), torque / 700);
  UI.stat("power", f1(power), power / 760);
  UI.stat("accel", f1(Math.max(car.accel, 0) / 9.81, 2), Math.max(car.accel, 0) / 9.81 / 1.2);
  UI.stat("split", s.bb + " / " + (100 - s.bb) + "  ·  " + f1((decel * M.MASS * (s.bb / 100)) / 1000, 1) + " kN front");
  UI.say("gear " + (detached() ? "—" : car.gear) + " · " + f1(car.v * 3.6) + " km/h · " + f1(sim.rpm / 1000, 1) + "k rpm");
}

let described = "";
function describe() {
  const t = detached() ? "Steering wheel removed. The gearbox cannot be shifted." : "Gear " + car.gear + " of 8 at " + Math.round(car.v * 3.6) + " kilometres per hour, " + Math.round(sim.rpm) + " revs.";
  if (t !== described) {
    described = t;
    UI.stage.setAttribute("aria-label", t);
  }
}

function doShift(direction) {
  if (detached()) return false;
  const ok = M.shift(car, direction, sim.now);
  if (ok) {
    if (direction > 0) sim.padU = 1;
    else sim.padD = 1;
  }
  return ok;
}
UI.action("up", () => doShift(1));
UI.action("down", () => doShift(-1));
UI.action("launch", () => {
  car.v = 0.5;
  car.gear = 1;
  sim.rpm = 3000;
});
UI.action("stop", () => {
  car.v = 0;
  car.gear = 1;
  sim.rpm = 1500;
});

let ticks = 0;
function tick(dt, calm, now) {
  sim.now = now;
  const s = UI.state;
  const mode = M.MODES[MODES.indexOf(s.mode)];
  const throttle = s.throttle / 100;
  const brake = s.brake / 100;
  M.step(car, throttle, brake, mode, dt, now);
  // a slipping clutch lets the engine rev freely at a standstill
  let rpmTarget = car.rpm;
  if (car.v < 6) rpmTarget = Math.max(car.rpm, 1500 + throttle * 9500 * (1 - car.v / 12));
  if (car.limiter) rpmTarget = M.REV_LIMIT + Math.sin(now * 90) * 220;
  sim.rpm = calm ? rpmTarget : follow(sim.rpm, rpmTarget, dt, car.limiter ? 0.015 : 0.06, false);
  if (s.auto && !detached()) {
    if (car.rpm > 14150 && car.gear < 8 && now - car.shiftAt > 0.4) doShift(1);
    else if (car.rpm < 7600 && car.gear > 1 && now - car.shiftAt > 0.5 && car.v > 4) doShift(-1);
  }
  sim.detach = follow(sim.detach, s.detach ? 1 : 0, dt, 0.28, calm);
  // the drum turns to the new gear; rings and forks follow its grooves
  sim.drum = calm ? car.gear : follow(sim.drum, car.gear, dt, 0.075, false);
  if (Math.abs(sim.drum - car.gear) < 0.002) sim.drum = car.gear;
  sim.padU = follow(sim.padU, 0, dt, 0.16, calm);
  sim.padD = follow(sim.padD, 0, dt, 0.16, calm);
  if (sim.padU < 0.01) sim.padU = 0;
  if (sim.padD < 0.01) sim.padD = 0;
  // the displayed turning speed is slowed so that it can be followed
  const shown = 0.35 + 7.2 * Math.pow(Math.max(0, sim.rpm - 1000) / (M.REV_LIMIT - 1000), 0.7);
  sim.phase += shown * dt;
  sim.shaft += (shown / M.pairRatio(car.gear)) * dt;
  const p = pose();
  paint(p);
  if (ticks++ % 3 === 0) {
    stats();
    describe();
  }
  const idle = car.v < 0.05 && throttle === 0 && brake === 0 && sim.padU === 0 && sim.padD === 0 && Math.abs(sim.drum - car.gear) < 0.01 && Math.abs(sim.detach - (s.detach ? 1 : 0)) < 0.002 && sim.rpm < 1600;
  return !idle || UI.playing;
}

/* click the stage: paddles shift, switches step round */
function viewPoint(e) {
  const svg = UI.svg;
  const pt = svg.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  const m = svg.getScreenCTM();
  return m ? pt.matrixTransform(m.inverse()) : { x: 0, y: 0 };
}
const hitAt = (e) => {
  const v = viewPoint(e);
  return GEO.hotspots(P, sim.detach).find((h) => Math.hypot(h.at[0] - v.x, h.at[1] - v.y) < h.r);
};
UI.stage.addEventListener("pointerdown", (e) => {
  const hit = hitAt(e);
  if (!hit) return;
  UI.stop();
  if (hit.id === "up") doShift(1);
  else if (hit.id === "down") doShift(-1);
  else if (hit.id === "bb") UI.set("bb", UI.state.bb >= 60 ? 54 : UI.state.bb + 1, "user");
  else if (hit.id === "diff") UI.set("diff", DIFFS[(DIFFS.indexOf(UI.state.diff) + 1) % 3], "user");
  else if (hit.id === "mode") UI.set("mode", MODES[(MODES.indexOf(UI.state.mode) + 1) % 3], "user");
  UI.wake();
});
UI.stage.addEventListener("pointermove", (e) => {
  UI.stage.style.cursor = hitAt(e) ? "pointer" : "";
});
UI.stage.addEventListener("keydown", (e) => {
  if (e.key === "ArrowUp" || e.key === "PageUp" || e.key === "+" || e.key === "=") {
    e.preventDefault();
    UI.stop();
    doShift(1);
    UI.wake();
  }
  if (e.key === "ArrowDown" || e.key === "PageDown" || e.key === "-") {
    e.preventDefault();
    UI.stop();
    doShift(-1);
    UI.wake();
  }
});

UI.loop(tick);
UI.on(() => UI.wake());
paint(pose());
stats();

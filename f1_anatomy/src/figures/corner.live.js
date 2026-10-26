// Live behaviour of the corner figure. Runs in the page; MODEL, GEO, UI, FA, DATA, k, G, T are in scope.
const P = DATA.P;
const items = GEO.items(P);
const nodes = {};
UI.$$("[data-item]").forEach((el) => (nodes[el.dataset.item] = el));

const car = MODEL.makeCar();
const sim = { ghost: 0, v: 0, temp: 330, spin: 0, kerbAt: -10, padTarget: 0, pad: 0, key: "", lo: null, extra: 0 };
const KERB_HEIGHT = () => UI.state.kerb / 1000;
const lo = (pedal, lat, v) => MODEL.loads(pedal, lat, v);

// start from a car that has settled at rest on the pad
sim.lo = lo(0, 0, 0);
MODEL.settleCar(car, 0);
sim.extra = 0;

const mm = (m) => (m * 1000).toFixed(0);
const celsius = (K) => K - 273.15;
const lastKeys = {};
let lastPose = null;

function pose() {
  const L = sim.lo;
  const roll = MODEL.CORNER.rollingRadius;
  const omega = sim.v / roll;
  return GEO.poseOf({
    xs: car.xs,
    xu: car.xu,
    zr: sim.pad,
    spin: sim.spin,
    temp: sim.temp,
    usage: L.usage,
    fxs: L.fx / L.limit,
    fys: L.fy / L.limit,
    patch: MODEL.patchLength(car.force),
    glowAmount: MODEL.glow(sim.temp),
    flow: omega,
    ghost: sim.ghost,
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
    if (item.cls === "fa-rubber") {
      const o = (1 - 0.9 * p.ghost).toFixed(3);
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

function stats(p) {
  const L = sim.lo;
  UI.stat("speed", (sim.v * 3.6).toFixed(0));
  UI.stat("load", (car.force / 1000).toFixed(2), car.force / 10000);
  UI.stat("grip", Math.round(L.usage * 100) + "", L.usage);
  UI.stat("travel", mm(car.travel), car.travel / 0.05);
  UI.stat("tyre", (car.tyreDeflection * 1000).toFixed(1) + " / " + mm(MODEL.patchLength(car.force)), car.tyreDeflection / 0.03);
  UI.stat("disc", celsius(sim.temp).toFixed(0), (sim.temp - 300) / 1100);
  UI.stat("decel", (L.decel / 9.81).toFixed(1), L.decel / (6 * 9.81));
  UI.say((sim.v * 3.6).toFixed(0) + " km/h · load " + (car.force / 1000).toFixed(1) + " kN · disc " + celsius(sim.temp).toFixed(0) + " °C");
}

let described = "";
function describe() {
  const text = "Corner at " + Math.round(sim.v * 3.6) + " kilometres per hour. Wheel load " + (car.force / 1000).toFixed(1) + " kilonewtons, disc " + Math.round(celsius(sim.temp)) + " degrees Celsius.";
  if (text !== described) { described = text; UI.stage.setAttribute("aria-label", text); }
}

let ticks = 0;
UI.action("strike", () => { sim.kerbAt = performance.now() / 1000; });
UI.action("cool", () => { sim.temp = 330; });

function tick(dt, calm, now) {
  const s = UI.state;
  const target = s.speed / 3.6;
  const pedal = s.brake / 100;
  const L0 = lo(pedal, s.lat, Math.max(0.1, sim.v));
  // the slider is the speed; the brake takes it down, and the slider follows the car
  if (pedal > 0) {
    sim.v = Math.max(0, sim.v - L0.decel * dt);
    if (Math.abs(s.speed - sim.v * 3.6) > 0.4) UI.set("speed", sim.v * 3.6, "sim");
  } else {
    sim.v = calm ? target : follow(sim.v, target, dt, 0.6, false);
    if (Math.abs(sim.v - target) < 0.02) sim.v = target;
  }
  const L = (sim.lo = lo(pedal, s.lat, sim.v));
  sim.extra = L.load - L.staticLoad;

  // the road: kerb strike plus whatever the reader is pushing
  sim.pad = MODEL.kerb(now - sim.kerbAt, KERB_HEIGHT()) + (sim.user || 0); // the pad is a stiff ram: it follows the command directly
  MODEL.step(car, sim.pad, sim.extra, dt);

  // disc heat, in small steps so the fast end of the curve is stable
  const power = MODEL.brakingPower(L.fx, sim.v);
  const n = 4;
  for (let i = 0; i < n; i++) sim.temp = Math.max(300, sim.temp + MODEL.discRate(sim.temp, power, sim.v) * (dt / n));
  const omega = sim.v / MODEL.CORNER.rollingRadius;
  const shown = omega > 0 ? 0.4 + 6.8 * Math.pow(omega / 252, 0.6) : 0;
  sim.spin += shown * dt;
  sim.ghost = follow(sim.ghost, s.ghost ? 1 : 0, dt, 0.35, calm);

  const p = pose();
  lastPose = p;
  paint(p);
  if (ticks++ % 3 === 0) { stats(p); describe(); }
  const settled = Math.abs(sim.ghost - (s.ghost ? 1 : 0)) < 0.002 && sim.v === target && pedal === 0 && Math.abs(car.vs) < 0.002 && Math.abs(car.vu) < 0.002 && sim.temp < 332 && now - sim.kerbAt > 1.2 && !sim.user;
  return !settled || UI.playing;
}

// dragging anywhere on the figure pushes the road up and down
let drag = null;
UI.stage.addEventListener("pointerdown", (e) => {
  drag = { y: e.clientY, id: e.pointerId };
  UI.stage.setPointerCapture(e.pointerId);
  UI.stop();
  UI.wake();
});
UI.stage.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const box = UI.svg.getBoundingClientRect();
  const dy = (drag.y - e.clientY) / (box.height / DATA.height);
  sim.user = Math.max(0, Math.min(0.06, dy * 0.00045));
  UI.wake();
});
const release = () => { drag = null; sim.user = 0; UI.wake(); };
UI.stage.addEventListener("pointerup", release);
UI.stage.addEventListener("pointercancel", release);
UI.stage.addEventListener("keydown", (e) => {
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); sim.kerbAt = performance.now() / 1000; UI.wake(); }
});

UI.loop(tick);
UI.on((id) => { UI.wake(); });
paint(pose());
stats(pose());

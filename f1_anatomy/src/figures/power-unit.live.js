// Live behaviour of the power unit figure. MODEL, GEO, UI, FA, DATA, k, G, T are in scope.
const P = DATA.P;
const items = GEO.items(P);
const nodes = {};
UI.$$("[data-item]").forEach((el) => (nodes[el.dataset.item] = el));

const st = MODEL.makeState();
st.rpm = UI.state.rpm;
const sim = { crank: 30, turbo: 0, last: null, soc: 0.5 };
st.socMJ = (UI.state.soc / 100) * MODEL.PU.storeWindow;
const lastKeys = {};
const sat = (v) => Math.max(-1, Math.min(1, v));

const INFO = {
  ice: "Internal combustion engine: 1.6 L V6, up to 15,000 rpm, fuel flow limited to 100 kg/h. Pistons fire about every 120 degrees, unevenly (90 and 150 degrees).",
  turbo: "Turbocharger: exhaust gas spins the turbine, which drives the compressor on the same shaft. Boost lets the engine burn the fuel it is allowed.",
  mguh: "MGU-H: a motor-generator on the turbo shaft. It turns spare exhaust energy into electricity, or spins the turbo up to remove lag.",
  mguk: "MGU-K: a motor-generator on the crankshaft. It adds up to 120 kW, or recovers up to 120 kW as brake energy.",
  es: "Energy store: the battery. About 4 MJ may leave it each lap to drive the MGU-K, and 2 MJ may be recovered by the MGU-K.",
  ce: "Control electronics: they decide how every kilowatt moves between the store, the MGU-K and the MGU-H.",
  intake: "Intake: compressed air is cooled in the intercooler, then fills the plenum that feeds the six cylinders.",
  exhaust: "Exhaust: the six exhaust runners collect into the turbine. This hot gas is the energy the MGU-H recovers.",
  rig: "Dyno bench: the engine drives the dynamometer, which measures its shaft power.",
};
const GROUP_FOCUS = { ice: ["ice"], turbo: ["turbo"], mguh: ["mguh"], mguk: ["mguk"], es: ["es"], ce: ["ce"], intake: ["intake"], exhaust: ["exhaust"], rig: [] };

function inputs() {
  const s = UI.state;
  return { rpm: s.rpm, throttle: s.throttle / 100, brake: s.brake / 100, mode: s.mode };
}

function pose(b) {
  return GEO.poseOf({
    crank: sim.crank,
    turbo: sim.turbo,
    load: Math.min(1, b.thr * 1.15),
    exh: Math.round(Math.min(1, b.turbKW / 150) * 20) / 20,
    air: Math.round(Math.min(1, b.air / 0.5) * 20) / 20,
    k: Math.round(sat(b.kW / 120) * 20) / 20,
    h: Math.round(sat(b.hKW / 80) * 20) / 20,
    es: Math.round(sat(b.esKW / 120) * 20) / 20,
    soc: Math.round((st.socMJ / MODEL.PU.storeWindow) * 16) / 16,
  });
}

function paint(p) {
  items.forEach((item) => {
    if (!item.live) return;
    const el = nodes[item.id];
    if (!el) return;
    const key = (item.deps || []).map((d) => (typeof p[d] === "number" ? p[d].toFixed(item.fine ?? (d === "crank" ? 1 : 2)) : p[d])).join("|");
    if (lastKeys[item.id] !== key) {
      lastKeys[item.id] = key;
      el.innerHTML = item.live(p);
    }
  });
}

const signed = (v, d = 0) => (v > 0.5 ? "+" : "") + v.toFixed(d);
function stats(b) {
  UI.stat("rpm", Math.round(st.rpm).toLocaleString("en-US"), st.rpm / 15000);
  UI.stat("ice", b.iceKW.toFixed(0), b.iceKW / 650);
  UI.stat("mguk", signed(b.kW), Math.abs(b.kW) / 120);
  UI.stat("mguh", signed(b.hKW), Math.abs(b.hKW) / 90);
  UI.stat("total", b.totalKW.toFixed(0) + " / " + Math.round(b.totalHp).toLocaleString("en-US"), b.totalKW / 780);
  UI.stat("boost", b.boost.toFixed(1), (b.boost - 1) / 2.7);
  UI.stat("soc", Math.round((st.socMJ / MODEL.PU.storeWindow) * 100) + "", st.socMJ / MODEL.PU.storeWindow);
  UI.stat("fuel", b.flow.toFixed(0), b.flow / 100);
  UI.stat("deploy", st.deployed.toFixed(1) + " / " + MODEL.PU.deployPerLap, st.deployed / MODEL.PU.deployPerLap);
  UI.stat("harvest", st.harvested.toFixed(1) + " / " + MODEL.PU.harvestPerLap, st.harvested / MODEL.PU.harvestPerLap);
  UI.say(Math.round(st.rpm / 100) * 100 + " rpm · " + Math.round(b.totalHp) + " hp · boost " + b.boost.toFixed(1) + " bar");
}

let described = "";
function describe(b) {
  const text = "Power unit at " + Math.round(st.rpm / 100) * 100 + " revs per minute, " + Math.round(b.totalHp) + " horsepower in total, battery " + Math.round((st.socMJ / MODEL.PU.storeWindow) * 100) + " percent.";
  if (text !== described) { described = text; UI.stage.setAttribute("aria-label", text); }
}

UI.action("lap", () => { MODEL.newLap(st); });
UI.on((id, value, by) => {
  if (id === "soc" && by !== "sim") st.socMJ = (value / 100) * MODEL.PU.storeWindow;
  UI.wake();
});

let ticks = 0;
function tick(dt, calm) {
  const slow = UI.state.slow ? 0.18 : 1;
  const inp = inputs();
  const b = MODEL.step(st, inp, dt);
  // the drawing turns slower than the real thing: 250 revolutions a second cannot be followed
  const revPerSec = (st.rpm / 15000) * 1.5 * slow;
  if (!calm) sim.crank += revPerSec * 360 * dt;
  const turboPerSec = (st.turbo / MODEL.PU.turboMax) * 3.6 * slow;
  if (!calm) sim.turbo += turboPerSec * Math.PI * 2 * dt;
  // the store slider shows the real charge, wherever the car puts it
  const socPct = (st.socMJ / MODEL.PU.storeWindow) * 100;
  if (Math.abs(UI.state.soc - socPct) > 0.4) UI.set("soc", socPct, "sim");
  const p = pose(b);
  paint(p);
  if (ticks++ % 3 === 0) { stats(b); describe(b); }
  return true;
}

// drag across the figure to change the revs, click a part to ask what it is
let drag = null;
UI.stage.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, rpm: UI.state.rpm, moved: false, target: e.target, id: e.pointerId };
});
UI.stage.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x;
  if (!drag.moved && Math.abs(dx) < 5) return;
  if (!drag.moved) { drag.moved = true; UI.stop(); UI.stage.setPointerCapture(drag.id); }
  const box = UI.svg.getBoundingClientRect();
  UI.set("rpm", Math.round((drag.rpm + (dx / box.width) * 11000) / 100) * 100, "user");
});
UI.stage.addEventListener("pointerup", (e) => {
  if (drag && !drag.moved) {
    const g = drag.target.closest && drag.target.closest("[data-group]");
    const name = g && g.dataset.group;
    if (name && INFO[name]) { UI.stop(); UI.focus(GROUP_FOCUS[name] || []); UI.say(INFO[name].split(":")[0]); UI.stage.setAttribute("aria-description", INFO[name]); window.__puInfo = INFO[name]; showInfo(INFO[name]); }
  }
  drag = null;
});
UI.stage.addEventListener("pointercancel", () => { drag = null; });
UI.stage.addEventListener("keydown", (e) => {
  const step = e.shiftKey ? 1000 : 500;
  if (e.key === "ArrowUp" || e.key === "ArrowRight") { e.preventDefault(); UI.stop(); UI.set("rpm", UI.state.rpm + step, "user"); }
  if (e.key === "ArrowDown" || e.key === "ArrowLeft") { e.preventDefault(); UI.stop(); UI.set("rpm", UI.state.rpm - step, "user"); }
});

const noteEl = UI.$(".fa-note");
const noteText = noteEl ? noteEl.textContent : "";
function showInfo(text) {
  if (noteEl) noteEl.textContent = text + "  ·  " + noteText;
}

UI.loop(tick);
const b0 = MODEL.balance({ ...inputs(), socMJ: st.socMJ, deployed: 0, harvested: 0 }, st.turbo);
paint(pose(b0));
stats(b0);

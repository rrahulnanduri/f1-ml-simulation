import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as GEO from "./drivetrain.geo.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const meta = {
  id: "drivetrain",
  num: 4,
  title: "Transmission and Steering",
  blurb: "An eight-speed sequential gearbox in cutaway, and the detachable steering wheel that drives it: paddles, rotaries and a screen.",
  tryit: "Pull the paddles · turn the switches · take the wheel off",
};

export function build() {
  const P = GEO.projection();
  const list = GEO.items(P);
  const pose = GEO.REST_POSE();
  const body = list.map((it) => {
    const [dx, dy, dz] = GEO.shiftOf(it, pose);
    const tr = it.xform ? ` transform="${k.translateAlong(P, [dx, dy, dz])}"` : "";
    const cls = it.cls ? ` class="${it.cls}"` : "";
    return `<g data-item="${it.id}" data-group="${it.group || ""}"${cls}${tr}>${GEO.markup(it, pose)}</g>`;
  });
  return {
    id: meta.id,
    title: meta.title,
    kicker: "Fig 4 of 5",
    fig: "Fig 4",
    plateTitle: "Gearbox and wheel",
    hint: "Click the paddles or press ↑ ↓",
    readout: "gear 1 · 0 km/h · 4.0k rpm",
    intro:
      "Cars use an eight-speed semi-automatic sequential gearbox controlled by paddle shifters. A lightweight, detachable steering wheel manages gear changes, brake balance and engine mappings. Here the gearbox is cut open on a bench next to its wheel.",
    width: GEO.WIDTH,
    height: GEO.HEIGHT,
    label:
      "An eight-speed Formula 1 gearbox cut open from above on a bench. Two parallel shafts carry eight pairs of gears. Four dog rings on the lower shaft slide between neighbouring gears to lock one of them to the shaft, moved by four forks that follow grooves on a rotating shift drum. To the left stands the detachable steering wheel with a screen, paddles and rotary switches.",
    svg: body,
    stageLabel: "Gearbox and steering wheel. Press the up and down arrow keys to shift.",
    keys: [
      { mark: "lit", label: "The gear that is locked to its shaft, its dog ring and fork" },
      { mark: "raised", label: "Shafts, gears and the shift drum" },
      { mark: "flat", label: "Steering wheel, paddles and rotary switches" },
    ],
    caption:
      "Eight pairs of gears are always meshed; a dog ring slides to lock one pair to the layshaft. The forks follow grooves in the shift drum, so one paddle pull turns the drum an eighth of a turn. Gear sizes are drawn from the real ratios; the drawing turns slower than the real shafts.",
    controls: [
      { type: "range", id: "throttle", label: "Throttle", min: 0, max: 100, step: 1, value: 0, unit: "%", help: "From a standstill the clutch slips so the engine can rev freely." },
      { type: "range", id: "brake", label: "Brake pedal", min: 0, max: 100, step: 1, value: 0, unit: "%" },
      { type: "button", id: "down", label: "◀ Downshift paddle" },
      { type: "button", id: "up", label: "Upshift paddle ▶", primary: true },
      { type: "toggle", id: "auto", label: "Shift for me", value: true, help: "Upshifts near 14,000 rpm and downshifts when the engine drops under about 7,500." },
      { type: "range", id: "bb", label: "Brake balance (to the front)", min: 54, max: 60, step: 1, value: 58, unit: "%" },
      { type: "choice", id: "diff", label: "Differential", value: "mid", options: [{ v: "open", label: "Open" }, { v: "mid", label: "Mid" }, { v: "locked", label: "Locked" }] },
      { type: "choice", id: "mode", label: "Engine mode", value: "quali", options: [{ v: "save", label: "Save" }, { v: "race", label: "Race" }, { v: "quali", label: "Quali" }] },
      { type: "toggle", id: "detach", label: "Take the wheel off", value: false, help: "With the quick release open the wheel slides off and the gearbox cannot be shifted." },
    ],
    stats: [
      { id: "speed", label: "Speed", unit: "km/h", bar: true },
      { id: "rpm", label: "Engine", unit: "rpm", bar: true },
      { id: "gear", label: "Gear", unit: "", bar: true },
      { id: "ratio", label: "Overall ratio", unit: "" },
      { id: "torque", label: "Crank torque", unit: "N m", bar: true },
      { id: "power", label: "Power", unit: "kW", bar: true },
      { id: "accel", label: "Acceleration", unit: "g", bar: true },
      { id: "split", label: "Brake split front / rear", unit: "" },
    ],
    steps: [
      { title: "Eight pairs of gears, always meshed", text: "The engine drives the upper shaft, and every gear on it is fixed. Each one meshes with a gear on the lower shaft that spins freely, until a dog ring locks one pair. First gear is the big pair, eighth the nearly equal one.", set: { throttle: 100, brake: 0, auto: true, detach: false, bb: 58, diff: "mid", mode: "quali" }, focus: ["lay"], dwell: 11 },
      { title: "A paddle pull turns a drum", text: "One tap turns the shift drum an eighth of a turn. Its grooves slide a fork, the fork slides a dog ring, and the ring locks the next gear to the shaft. In the real thing this takes about 28 milliseconds, and drive is barely interrupted.", set: { throttle: 55, auto: false }, run: "stop", then: [{ after: 0.6, run: "launch" }, { after: 2.2, run: "up" }, { after: 4.2, run: "up" }, { after: 6.4, run: "down" }], focus: ["shift"], dwell: 9 },
      { title: "Why eight", text: "Each gear lets the engine work between about 9,000 and 15,000 rpm over a different slice of road speed. First tops out near 85 km/h, eighth near 350 km/h.", set: { throttle: 100, auto: true }, focus: ["lay", "main"], dwell: 11 },
      { title: "The wheel is the control room", text: "The rotary switches set brake balance, the differential and the engine mode, and the screen shows gear, speed and rev lights. Turn them from here or click them on the wheel.", set: { throttle: 40, auto: true }, then: [{ after: 1.4, set: { bb: 55 } }, { after: 2.6, set: { diff: "locked" } }, { after: 3.8, set: { mode: "race" } }, { after: 5.2, set: { bb: 59, mode: "quali", diff: "mid" } }], focus: ["wheel"], dwell: 8 },
      { title: "Brake balance", text: "More brake to the front stops the car harder but can lock the front tyres; more to the rear is stable until the rears lock. The driver trims it corner by corner.", set: { throttle: 100, auto: true, brake: 0 }, then: [{ after: 3.5, set: { throttle: 0, brake: 70 } }, { after: 6, set: { brake: 0 } }], focus: ["wheel"], dwell: 9 },
      { title: "Take the wheel off", text: "The wheel is on a quick release so the driver can get in and out. With it off the car cannot be shifted: the screen goes blank and the paddles do nothing.", set: { throttle: 0, brake: 0, detach: true }, then: [{ after: 4.5, set: { detach: false } }], focus: ["wheel"], dwell: 8 },
    ],
    note: "A simplified drivetrain: rounded ratios and a smooth power curve, chosen so first gear tops out near 85 km/h and eighth near 350 km/h.",
    data: { P },
    modules: [
      { name: "M", file: join(HERE, "drivetrain.model.mjs") },
      { name: "GEO", file: join(HERE, "drivetrain.geo.mjs") },
    ],
    live: readFileSync(join(HERE, "drivetrain.live.js"), "utf8"),
  };
}

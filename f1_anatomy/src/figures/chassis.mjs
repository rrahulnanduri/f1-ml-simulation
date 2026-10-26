import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as GEO from "./chassis.geo.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const meta = {
  id: "chassis",
  num: 1,
  title: "Chassis and Safety",
  blurb: "The carbon monocoque on a crash-test rig: crush the nose, load the Halo, hit the side, and watch the survival cell stay whole.",
  tryit: "Run a 15 m/s crash · load the Halo · fade the carbon skin",
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
    kicker: "Fig 1 of 5",
    fig: "Fig 1",
    plateTitle: "Survival cell test rig",
    hint: "Drag up and down to change the test",
    readout: "15.0 m/s · nose 150 kN · ready",
    intro:
      "The core is the monocoque, a lightweight carbon fibre shell that protects the driver. It carries the Halo for head protection and houses critical components like the fuel tank and the electronics. This rig puts the cell on a sled so you can crush its nose, load its Halo and hit its side.",
    width: GEO.WIDTH,
    height: GEO.HEIGHT,
    label:
      "A crash-test rig. A Formula 1 survival cell, cut away on its near side, sits on a sled on two rails. Inside are a dummy driver in a seat, the fuel cell behind the seat, a battery and electronics bay ahead of the legs, and the pedal box. A titanium Halo arches over the cockpit on three mounts. A crushable nose points at a concrete wall. A hydraulic ram stands over the Halo and another pushes on the side. A trace of deceleration against time is drawn on the base.",
    svg: body,
    stageLabel: "Crash test rig. Drag up and down to change the speed or the load, or press space to run the test.",
    keys: [
      { mark: "lit", label: "Crushing nose cap and load rams" },
      { mark: "raised", label: "Carbon cell, fuel cell, battery and electronics" },
      { mark: "dotted", label: "Deceleration trace with the 60 g and 25 g limits" },
    ],
    caption:
      "The near wall of the tub is faded so you can see what is inside. Left to right on the base plate is time; away from you is deceleration. The car stops over about 0.06 s; the animation is slowed down to follow it.",
    controls: [
      { type: "choice", id: "test", label: "Test", value: "frontal", options: [{ v: "frontal", label: "Frontal impact" }, { v: "side", label: "Side impact" }, { v: "halo", label: "Halo load" }] },
      { type: "range", id: "speed", label: "Impact speed", min: 2, max: 15, step: 0.5, value: 15, decimals: 1, unit: "m/s", help: "The frontal test is 15 m/s. The side test only goes to 10." },
      { type: "range", id: "nose", label: "Nose crush force", min: 90, max: 270, step: 5, value: 150, unit: "kN", help: "Too soft and the nose is used up; too stiff and the driver feels it." },
      { type: "range", id: "load", label: "Halo load", min: 20, max: 125, step: 1, value: 116, unit: "kN" },
      { type: "range", id: "skin", label: "Carbon skin", min: 5, max: 100, step: 1, value: 22, unit: "%", help: "Fade the near wall away to see inside the cell." },
      { type: "button", id: "run", label: "Run the test", primary: true },
    ],
    stats: [
      { id: "peak", label: "Peak deceleration", unit: "g", bar: true },
      { id: "avg", label: "Average deceleration", unit: "g", bar: true },
      { id: "crush", label: "Structure crushed", unit: "mm", bar: true },
      { id: "energy", label: "Energy absorbed", unit: "kJ", bar: true },
      { id: "intrusion", label: "Intrusion into the cell", unit: "mm", bar: true },
      { id: "head", label: "Driver's head travel", unit: "mm", bar: true },
      { id: "halo", label: "Halo load / deflection", unit: "kN · mm" },
      { id: "result", label: "Result", unit: "" },
    ],
    steps: [
      { title: "A carbon shell around the driver", text: "The monocoque is one moulded carbon-fibre tub, light enough to lift by hand and strong enough to survive a crash. Everything else bolts to it.", set: { test: "frontal", skin: 100, speed: 15, nose: 150 }, focus: ["shell"], dwell: 6 },
      { title: "The driver sits low inside it", text: "Fade the skin and the driver is seated low, feet forward of the front axle line, held by a six-point harness and a head-and-neck restraint.", set: { skin: 14 }, focus: ["driver"], dwell: 6 },
      { title: "Fuel and electronics live in the tub", text: "Behind the seat is the rubber-bladder fuel cell. Ahead of the legs sit the battery and the control electronics, kept inside the strongest part of the car.", focus: ["fuel", "electronics"], dwell: 7 },
      { title: "The Halo", text: "A titanium hoop on three mounts. It is tested with loads of the order of 100 kN on its apex, and bends only a few millimetres.", set: { test: "halo", load: 116, skin: 22 }, run: "run", focus: ["halo"], dwell: 9 },
      { title: "Crush the nose, not the cell", text: "At 15 m/s the nose structure crumples front to back, turning about 88 kJ into deformation over about half a metre. The cell, and the driver in it, stop gently.", set: { test: "frontal", speed: 15, nose: 150, skin: 22 }, run: "run", focus: ["crash", "plot"], dwell: 10 },
      { title: "Too soft, too stiff", text: "Soften the nose and it bottoms out: the cell is hit. Stiffen it and the deceleration spikes past the 60 g limit. The structure has to be just right.", set: { nose: 100 }, run: "run", then: [{ after: 5.6, set: { nose: 260 }, run: "run" }], focus: ["crash", "plot"], dwell: 12 },
      { title: "The side is protected too", text: "Side crash structures take a ram at up to 10 m/s. Beyond what they can crush, the cell wall itself starts to dent.", set: { test: "side", speed: 7.5, nose: 150 }, run: "run", then: [{ after: 5.4, set: { speed: 10 }, run: "run" }], focus: ["side"], dwell: 11 },
    ],
    note: "A simplified model with rounded, typical figures in the spirit of the FIA tests. Real tests have more requirements than these.",
    data: { P, height: GEO.HEIGHT },
    modules: [
      { name: "MODEL", file: join(HERE, "chassis.model.mjs") },
      { name: "GEO", file: join(HERE, "chassis.geo.mjs") },
    ],
    live: readFileSync(join(HERE, "chassis.live.js"), "utf8"),
  };
}

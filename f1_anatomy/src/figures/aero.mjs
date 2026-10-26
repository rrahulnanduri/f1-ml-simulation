import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as GEO from "./aero.geo.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const meta = {
  id: "aero",
  num: 3,
  title: "Aerodynamics",
  blurb: "A car in a wind tunnel on a rolling road: wings, floor and diffuser press it down, and the rear flap opens for DRS.",
  tryit: "Open the DRS flap · drop the ride height until the floor stalls",
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
    kicker: "Fig 3 of 5",
    fig: "Fig 3",
    plateTitle: "Wind tunnel",
    hint: "Click wing: DRS · drag: ride height",
    readout: "250 km/h · down 11 kN · drag 3.1 kN",
    intro:
      "Wings, the floor and the diffuser generate massive downforce that pushes the car onto the track. The Drag Reduction System (DRS) on the rear wing can open to cut drag for higher top speeds on the straights. This is the car in a wind tunnel, with smoke streamlines.",
    width: GEO.WIDTH,
    height: GEO.HEIGHT,
    label:
      "A simplified Formula 1 car on the rolling road of a wind tunnel, seen from ahead and to its left. Smoke streamlines run from a rake in front of the nose over the front wing, the cockpit and the airbox, and through the rear wing, whose upper flap can hinge open for DRS. Blue lines under the floor show the fast, low-pressure air through the floor and the diffuser. Arrows show where the downforce acts, and a dial gauge on the left reads the ride height.",
    svg: body,
    stageLabel: "Wind tunnel. Click the rear wing or press space for DRS, drag up and down to change the ride height.",
    keys: [
      { mark: "lit", label: "Rear flap (DRS) and front flap: the parts that move" },
      { mark: "blue", label: "Under the floor: fast, low-pressure air" },
      { mark: "raised", label: "Grey streamlines: air over the car" },
      { mark: "red", label: "Drag" },
    ],
    caption:
      "The car sits on a moving belt so the road moves at the same speed as the air. Black arrows are downforce from the front wing, floor, diffuser and rear wing; the red arrow is drag. Switch on the X-ray to see the air under the floor.",
    controls: [
      { type: "range", id: "speed", label: "Tunnel speed", min: 0, max: 340, step: 5, value: 250, unit: "km/h", help: "Downforce and drag both grow with the square of speed." },
      { type: "toggle", id: "drs", label: "DRS open", value: false, help: "Opens the rear wing's upper flap on a straight: less drag, a little less downforce." },
      { type: "range", id: "ride", label: "Ride height (plank above the road)", min: 20, max: 90, step: 1, value: 35, unit: "mm", help: "Low is better until the floor loses its seal and stalls, below about 24 mm." },
      { type: "range", id: "fw", label: "Front flap angle", min: 0, max: 30, step: 1, value: 15, unit: "°" },
      { type: "range", id: "rw", label: "Rear wing angle", min: 10, max: 35, step: 1, value: 22, unit: "°" },
      { type: "toggle", id: "xray", label: "X-ray the floor", value: false, help: "Fades the floor so the air under it shows." },
    ],
    stats: [
      { id: "down", label: "Downforce", unit: "kN", bar: true },
      { id: "split", label: "Front / rear", unit: "%" },
      { id: "drag", label: "Drag", unit: "kN", bar: true },
      { id: "ld", label: "Downforce per drag", unit: "×" },
      { id: "floor", label: "From floor and diffuser", unit: "%", bar: true },
      { id: "vmax", label: "Top speed, power vs drag", unit: "km/h" },
      { id: "gain", label: "DRS gain at the top", unit: "km/h" },
    ],
    steps: [
      { title: "The front wing reads the air first", text: "It makes about a fifth of the downforce and decides how the air reaches the rest of the car. A steeper flap presses harder, and adds drag.", set: { speed: 250, drs: false, xray: false, ride: 35, fw: 15, rw: 22 }, focus: ["frontwing"], dwell: 6.5 },
      { title: "The floor is the biggest wing", text: "Fade the car and look underneath. Air squeezed through the tunnels speeds up, its pressure drops, and the car is sucked onto the road. Nearly half the downforce comes from here.", set: { xray: true, ride: 35 }, focus: ["floor", "flow"], dwell: 7 },
      { title: "The diffuser lets the air go home", text: "The rising ramp at the back slows the fast air down gently, which keeps the floor's suction strong all the way along.", set: { xray: true }, focus: ["diffuser"], dwell: 6 },
      { title: "The rear wing balances the front", text: "A steeper rear wing adds downforce at the back, and a lot more drag. Teams pick the angle for each circuit.", set: { xray: false, rw: 28 }, then: [{ after: 3, set: { rw: 22 } }], focus: ["rearwing"], dwell: 7 },
      { title: "DRS on a straight", text: "The driver opens the rear flap. The slot lets air straight through, drag falls by about a fifth and the top speed rises. It closes again under braking.", set: { speed: 320, drs: true, rw: 22 }, then: [{ after: 5, set: { drs: false, speed: 250 } }], focus: ["rearwing", "forces"], dwell: 8.5 },
      { title: "Too low and the floor stalls", text: "Lower the car and the floor works harder, until the gap gets too small and the flow detaches. Downforce collapses, the car rises, the seal returns and it drops again: porpoising.", set: { speed: 280, ride: 20, xray: true, drs: false }, then: [{ after: 5.5, set: { ride: 35, xray: false } }], focus: ["floor", "gauge"], dwell: 8.5 },
    ],
    note: "A simplified model with rounded, typical values: about 16 kN of downforce at 300 km/h with the floor sealed, drag cut by about a fifth with DRS open. Top speed assumes 600 kW at the wheels.",
    data: { P, height: GEO.HEIGHT, width: GEO.WIDTH },
    modules: [
      { name: "MODEL", file: join(HERE, "aero.model.mjs") },
      { name: "GEO", file: join(HERE, "aero.geo.mjs") },
    ],
    live: readFileSync(join(HERE, "aero.live.js"), "utf8"),
  };
}

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as GEO from "./corner.geo.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const meta = {
  id: "corner",
  num: 5,
  title: "Suspension, Brakes and Tyres",
  blurb: "One front corner on a test rig: a push-rod suspension, a carbon brake disc and an 18-inch tyre, with a road you can push.",
  tryit: "Hit a kerb · brake from 300 km/h · watch the grip circle",
};

export function build() {
  const P = GEO.projection();
  const list = GEO.items(P);
  const pose = GEO.REST_POSE();
  const body = [
    GEO.windowDefs(P),
    ...list.map((it) => {
      const [dx, dy, dz] = GEO.shiftOf(it, pose);
      const tr = it.xform ? ` transform="${k.translateAlong(P, [dx, dy, dz])}"` : "";
      const clip = it.holed ? ' clip-path="url(#tyre-window)"' : "";
      const cls = it.cls ? ` class="${it.cls}"` : "";
      return `<g data-item="${it.id}" data-group="${it.group || ""}"${cls}${tr}${clip}>${GEO.markup(it, pose)}</g>`;
    }),
  ];

  return {
    id: meta.id,
    title: meta.title,
    kicker: "Fig 5 of 5",
    fig: "Fig 5",
    plateTitle: "Front-right corner",
    hint: "Drag up and down to push the road",
    readout: "0 km/h · load 1.8 kN · disc 57 °C",
    intro:
      "Push-rod or pull-rod suspension keeps the car stable and holds its aerodynamic ride height. Carbon-carbon brakes take extreme heat, and 18-inch tyres are the only physical contact with the track. This corner is a quarter of the car on a test rig: the road is the pad underneath.",
    width: GEO.WIDTH,
    height: GEO.HEIGHT,
    label:
      "A front corner of a Formula 1 car on a test rig, seen from inboard and ahead. A road pad on two rams sits under the tyre. Through the open end of the rim you see the carbon brake disc, its caliper and the hub. Two wishbones and a push rod join the upright to a stub of the chassis; the push rod turns a rocker that squeezes a damper and twists a torsion bar.",
    svg: body,
    stageLabel: "Suspension, brake and tyre corner. Drag up and down to push the road, or press space to hit a kerb.",
    keys: [
      { mark: "lit", label: "Push rod and damper: the load path" },
      { mark: "raised", label: "Wishbones, upright and rocker" },
      { mark: "flat", label: "Brake disc, caliper and hub inside the rim" },
    ],
    caption:
      "Seen from inboard and ahead. The rocker is mounted on the chassis stub; the push rod pushes it as the wheel rises, and it squeezes the damper and twists the torsion bar. The disc glows with its real temperature.",
    controls: [
      { type: "range", id: "speed", label: "Speed to run at", min: 0, max: 340, step: 5, value: 0, unit: "km/h", help: "Downforce grows with the square of speed and presses the car onto its tyres." },
      { type: "range", id: "brake", label: "Brake pedal", min: 0, max: 100, step: 1, value: 0, unit: "%", help: "Holding it slows the car. The disc turns the lost speed into heat." },
      { type: "range", id: "lat", label: "Cornering demand (outside wheel)", min: 0, max: 5, step: 0.1, value: 0, decimals: 1, unit: "g" },
      { type: "range", id: "kerb", label: "Kerb height", min: 5, max: 50, step: 1, value: 30, unit: "mm" },
      { type: "button", id: "strike", label: "Hit the kerb", primary: true },
      { type: "toggle", id: "ghost", label: "See through the tyre", value: false, help: "Fades the tyre away so the brake, hub and caliper inside the rim show." },
      { type: "button", id: "cool", label: "Cool the disc" },
    ],
    stats: [
      { id: "speed", label: "Speed", unit: "km/h" },
      { id: "load", label: "Wheel load", unit: "kN", bar: true },
      { id: "grip", label: "Grip used", unit: "%", bar: true },
      { id: "travel", label: "Suspension travel", unit: "mm", bar: true },
      { id: "tyre", label: "Tyre squash / patch length", unit: "mm", bar: true },
      { id: "disc", label: "Disc temperature", unit: "°C", bar: true },
      { id: "decel", label: "Braking", unit: "g", bar: true },
    ],
    steps: [
      { title: "The tyre is the only contact", text: "Everything the car does to the road goes through four patches of rubber, each about the size of a postcard. Standing still it carries about 1.8 kN.", set: { speed: 0, brake: 0, lat: 0, ghost: false }, focus: ["tyre", "pad"], dwell: 6 },
      { title: "Downforce squeezes the suspension", text: "At 300 km/h the wings press about 3.3 kN more onto this wheel. The push rod compresses, the car sits lower and the tyre squashes further.", set: { speed: 300, brake: 0, lat: 0 }, focus: ["suspension", "pushrod"], dwell: 6 },
      { title: "Push rod, rocker, damper", text: "A kerb throws the wheel up. The push rod turns the rocker, which squeezes the damper to take the energy and twists the torsion bar like a spring.", set: { speed: 300, kerb: 35 }, run: "strike", then: [{ after: 3.4, run: "strike" }], focus: ["pushrod", "damper", "torsion"], dwell: 7 },
      { title: "Brakes turn speed into heat", text: "Hard braking from 300 km/h pushes about a megawatt into one front disc. It climbs towards 1000 °C and glows, then cools in the airstream. The tyre is faded so you can see inside the rim.", set: { speed: 300, brake: 0, lat: 0, ghost: true }, then: [{ after: 1.8, set: { brake: 100 } }, { after: 5.6, set: { brake: 0 } }], focus: ["brake"], dwell: 10 },
      { title: "The grip circle", text: "A tyre has one budget of grip to share between braking and cornering. Ask for both and the dot hits the edge of the circle: the tyre is at its limit.", set: { speed: 200, brake: 0, lat: 3, ghost: false }, then: [{ after: 2, set: { brake: 45 } }, { after: 5.5, set: { brake: 0 } }], focus: ["tyre", "grip"], dwell: 8 },
    ],
    note: "A simplified quarter-car model with rounded, typical modern values. The wheel is drawn turning slower than it really would.",
    data: { P, height: GEO.HEIGHT },
    modules: [
      { name: "MODEL", file: join(HERE, "corner.model.mjs") },
      { name: "GEO", file: join(HERE, "corner.geo.mjs") },
    ],
    live: readFileSync(join(HERE, "corner.live.js"), "utf8"),
  };
}

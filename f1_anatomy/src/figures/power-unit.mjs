import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as GEO from "./power-unit.geo.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

export const meta = {
  id: "power-unit",
  num: 2,
  title: "Power Unit",
  blurb: "A 1.6-litre turbo V6 and two electric motors on a dyno bench, with every kilowatt of the 1,000 horsepower traced.",
  tryit: "Drag to rev · deploy and harvest · fill the lap budget",
};

export function build() {
  const P = GEO.projection();
  const list = GEO.items(P);
  const pose = GEO.REST_POSE();
  const body = list.map((it) => `<g data-item="${it.id}" data-group="${it.group || ""}">${GEO.markup(it, pose)}</g>`);

  return {
    id: meta.id,
    title: meta.title,
    kicker: "Fig 2 of 5",
    fig: "Fig 2",
    plateTitle: "Hybrid power unit",
    hint: "Drag across to rev · click a part",
    readout: "9,000 rpm · 1,023 hp · boost 3.6 bar",
    intro:
      "The hybrid power unit combines a 1.6-litre turbocharged V6 engine with electric motors. The system recovers energy during braking and from exhaust heat to boost performance to over 1,000 horsepower. This one is on a dyno bench, opened up so you can see where each kilowatt comes from.",
    width: GEO.WIDTH,
    height: GEO.HEIGHT,
    label:
      "A Formula 1 power unit on a dyno bench, seen from the side. A 90 degree V6 with its pistons, rods and crankshaft exposed, an electric motor-generator (MGU-K) at the front, a flywheel and shaft to a dynamometer at the back, and a turbocharger with a second motor-generator (MGU-H) between its compressor and turbine. The energy store and control electronics stand on the bench. Dashed lines show exhaust gas, intake air and electric power moving.",
    svg: body,
    stageLabel: "Power unit. Drag across to change the engine speed, or click a part to see what it does.",
    keys: [
      { mark: "red", label: "Hot exhaust gas" },
      { mark: "blue", label: "Intake air" },
      { mark: "lit", label: "Power to a motor" },
      { mark: "green", label: "Energy recovered" },
    ],
    caption:
      "Pistons, rods and crankshaft turn at a fraction of true speed. The glow in a cylinder is combustion, in this engine's real firing order. Dashes run in the direction energy moves.",
    controls: [
      { type: "range", id: "rpm", label: "Engine speed", min: 4000, max: 15000, step: 100, value: 9000, unit: "rpm", help: "The limit is 15,000 rpm. Fuel flow is capped at 100 kg/h above 10,500 rpm." },
      { type: "range", id: "throttle", label: "Throttle", min: 0, max: 100, step: 1, value: 100, unit: "%" },
      { type: "range", id: "brake", label: "Brake pedal", min: 0, max: 100, step: 1, value: 0, unit: "%", help: "Braking shuts the throttle and lets the MGU-K recover energy." },
      { type: "choice", id: "mode", label: "Energy mode", value: "deploy", options: [{ v: "deploy", label: "Deploy" }, { v: "balanced", label: "Balanced" }, { v: "harvest", label: "Harvest" }], help: "Deploy adds MGU-K power under throttle. Harvest charges the store even on the power." },
      { type: "range", id: "soc", label: "Battery charge", min: 0, max: 100, step: 1, value: 50, unit: "%", help: "Follows the real charge. Drag it to start from empty or full." },
      { type: "toggle", id: "slow", label: "Slow motion", value: false, help: "Slows the drawing so you can follow the firing order." },
      { type: "button", id: "lap", label: "Start a new lap" },
    ],
    stats: [
      { id: "rpm", label: "Engine speed", unit: "rpm", bar: true },
      { id: "ice", label: "Engine power", unit: "kW", bar: true },
      { id: "mguk", label: "MGU-K (+ drives)", unit: "kW", bar: true },
      { id: "mguh", label: "MGU-H (+ recovers)", unit: "kW", bar: true },
      { id: "total", label: "Total at the crank", unit: "kW / hp", bar: true },
      { id: "boost", label: "Boost, absolute", unit: "bar", bar: true },
      { id: "soc", label: "Battery charge", unit: "%", bar: true },
      { id: "fuel", label: "Fuel flow", unit: "kg/h", bar: true },
      { id: "deploy", label: "Deployed this lap", unit: "MJ", bar: true },
      { id: "harvest", label: "Recovered this lap", unit: "MJ", bar: true },
    ],
    steps: [
      { title: "A 1.6-litre V6 with a turbo", text: "Six cylinders in two banks of three, at 90 degrees. Watch the glow move from cylinder to cylinder: it fires about every 120 degrees of crank rotation, unevenly.", set: { rpm: 6500, throttle: 70, brake: 0, mode: "balanced", slow: true, soc: 60 }, run: "lap", focus: ["ice"], dwell: 7 },
      { title: "Exhaust heat becomes power", text: "Hot exhaust spins the turbine. The compressor on the same shaft needs part of that power for boost; the MGU-H takes the surplus as electricity.", set: { rpm: 11500, throttle: 100, brake: 0, mode: "balanced", slow: false }, run: "lap", focus: ["exhaust", "turbo", "mguh"], dwell: 7 },
      { title: "The MGU-K deploys", text: "On the throttle in deploy mode the MGU-K adds up to 120 kW straight to the crankshaft, from the store and from the MGU-H. Note the lap counter climbing.", set: { rpm: 12500, throttle: 100, brake: 0, mode: "deploy", slow: false, soc: 90 }, run: "lap", focus: ["mguk", "es", "ce"], dwell: 8 },
      { title: "Braking charges the battery", text: "Off the throttle and on the brakes, the MGU-K turns into a generator and recovers up to 120 kW, up to 2 MJ a lap. The green dashes now run into the store.", set: { rpm: 9000, throttle: 0, brake: 100, mode: "deploy", slow: false, soc: 30 }, run: "lap", focus: ["mguk", "es"], dwell: 8 },
      { title: "Over 1,000 horsepower", text: "Full throttle at 12,000 rpm and the engine alone makes about 640 kW. With 120 kW from the MGU-K the total passes 1,000 horsepower.", set: { rpm: 12000, throttle: 100, brake: 0, mode: "deploy", soc: 80 }, run: "lap", focus: ["ice", "turbo", "mguh", "mguk"], dwell: 8 },
      { title: "A budget per lap", text: "Only 4 MJ may leave the store each lap and 2 MJ may be recovered. Deploy flat out and the budget runs dry: the MGU-K stops adding power. Start a new lap to refill it.", set: { rpm: 12000, throttle: 100, brake: 0, mode: "deploy", soc: 100 }, run: "lap", dwell: 10 },
    ],
    note: "Simplified, regulation-era figures: about 50 percent thermal efficiency, 120 kW MGU-K, 4 MJ deploy and 2 MJ recovery per lap. The energy clock runs 8 times faster than the animation.",
    data: { P },
    modules: [
      { name: "MODEL", file: join(HERE, "power-unit.model.mjs") },
      { name: "GEO", file: join(HERE, "power-unit.geo.mjs") },
    ],
    live: readFileSync(join(HERE, "power-unit.live.js"), "utf8"),
  };
}

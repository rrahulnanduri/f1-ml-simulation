// Crash and load tests on the survival cell. Pure functions, SI units unless a name says otherwise.
// Simplified on purpose: rounded, typical figures in the spirit of the FIA tests, not a homologation model.

export const G = 9.81;

export const TEST = {
  mass: 780, // kg, test car and dummy
  frontalSpeed: 15, // m/s, the frontal test
  noseCrushable: 0.6, // m of nose structure that can be crushed before the cell is reached
  avgLimit: 25, // g, average deceleration the nose must stay under
  peakLimit: 60, // g, peak deceleration limit
  cellForce: 600e3, // N, what it takes to start denting the survival cell
  cellRate: 20e6, // N/m, how fast that grows with intrusion
  sideSpeedMax: 10, // m/s
  sideCrushable: 0.15, // m of side structure
  sideForce: 200e3, // N, crush force of the side structure
  haloStiffness: 11e6, // N/m at the apex, vertical
  haloLoadMax: 125e3, // N
  harnessK: 2.4e5, // N/m, driver restraint
  harnessC: 5000, // N s/m
  driver: 80, // kg, driver and seat share
};

/** Force profile of a progressive crush structure: a start-up peak settling to a plateau. */
export const profile = (x, plateau) => plateau * (1 + 0.9 * Math.exp(-x / 0.04));

/**
 * Run the frontal test: the car hits the wall at v with a nose of plateau force `forceN`.
 * Returns the trace (time s, deceleration g, crush m, driver forward travel m) and the summary.
 */
export function frontal(v, forceN, dt = 0.0002) {
  const m = TEST.mass;
  let x = 0;
  let vel = v;
  let t = 0;
  let r = 0;
  let rv = 0;
  let peak = 0;
  let headMax = 0;
  const trace = [];
  const xc = TEST.noseCrushable;
  let n = 0;
  while (vel > 0 && t < 0.6) {
    let F = profile(x, forceN);
    if (x > xc) F += TEST.cellForce + TEST.cellRate * (x - xc); // the nose is used up: the cell starts to take it
    const a = F / m;
    if (n++ % 10 === 0) trace.push({ t, g: a / G, x, r, v: vel });
    // the driver is held by the harness: relative to the car he is pushed forward by the car's deceleration
    const Fh = TEST.harnessK * r + TEST.harnessC * rv;
    const ra = a - Fh / TEST.driver;
    rv += ra * dt;
    r = Math.max(0, r + rv * dt);
    if (r === 0 && rv < 0) rv = 0;
    vel = Math.max(0, vel - a * dt);
    x += vel * dt;
    t += dt;
    peak = Math.max(peak, a / G);
    headMax = Math.max(headMax, r);
  }
  // let the driver settle back for the tail of the trace
  const intrusion = Math.max(0, x - xc);
  const crush = Math.min(x, xc);
  const energy = 0.5 * m * v * v;
  const avg = x > 0 ? (v * v) / (2 * x) / G : 0;
  const pass = peak <= TEST.peakLimit && (v < 12 || avg <= TEST.avgLimit) && intrusion < 0.005;
  return { v, forceN, d: x, crush, intrusion, peak, avg, energy, duration: t, head: headMax, pass, trace };
}

/** Side test: the ram hits the side structure at v. */
export function side(v, dt = 0.0002) {
  const m = TEST.mass;
  const xc = TEST.sideCrushable;
  let x = 0;
  let vel = v;
  let t = 0;
  let peak = 0;
  const trace = [];
  let n = 0;
  while (vel > 0 && t < 0.5) {
    let F = profile(x, TEST.sideForce);
    if (x > xc) F += TEST.cellForce + TEST.cellRate * (x - xc);
    const a = F / m;
    if (n++ % 10 === 0) trace.push({ t, g: a / G, x, r: 0, v: vel });
    vel = Math.max(0, vel - a * dt);
    x += vel * dt;
    t += dt;
    peak = Math.max(peak, a / G);
  }
  const intrusion = Math.max(0, x - xc);
  const energy = 0.5 * m * v * v;
  return { v, d: x, crush: Math.min(x, xc), intrusion, peak, avg: x > 0 ? (v * v) / (2 * x) / G : 0, energy, duration: t, head: 0, pass: intrusion < 0.005 && peak <= TEST.peakLimit, trace };
}

/** Halo load test: a load on the apex. Returns deflection, how the load shares between its mounts. */
export function halo(loadN) {
  const defl = loadN / TEST.haloStiffness;
  return { load: loadN, defl, front: loadN * 0.4, rear: loadN * 0.3, pass: loadN <= TEST.haloLoadMax };
}

/** Deceleration at time t of a stored trace (linear interpolation). */
export function traceAt(trace, t) {
  if (!trace.length) return { g: 0, x: 0, r: 0, v: 0 };
  if (t <= trace[0].t) return trace[0];
  for (let i = 1; i < trace.length; i++) {
    if (t <= trace[i].t) {
      const a = trace[i - 1];
      const b = trace[i];
      const f = (t - a.t) / (b.t - a.t);
      return { g: a.g + (b.g - a.g) * f, x: a.x + (b.x - a.x) * f, r: a.r + (b.r - a.r) * f, v: a.v + (b.v - a.v) * f };
    }
  }
  return trace[trace.length - 1];
}

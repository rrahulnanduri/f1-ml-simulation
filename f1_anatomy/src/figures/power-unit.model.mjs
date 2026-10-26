// Simplified power unit model: 1.6 L turbo V6 with MGU-K, MGU-H and an energy store.
// Rounded, regulation-era values. SI units unless a name says otherwise; power in kW, energy in MJ.

export const PU = {
  maxRpm: 15000,
  fuelCap: 100, // kg/h above 10,500 rpm
  lhv: 44.5e6, // J/kg
  etaPeak: 0.52, // thermal efficiency at full load
  kMax: 120, // kW, MGU-K limit
  deployPerLap: 4, // MJ the store may give to the MGU-K each lap
  harvestPerLap: 2, // MJ the MGU-K may harvest each lap
  storeWindow: 4, // MJ, the usable swing of the store shown here
  exhaustShare: 0.13, // of fuel power that the turbine can turn into shaft power
  timeLapse: 8, // the energy clock runs this many times faster than the animation
  turboMax: 125000, // rpm
};

export const maxFlow = (rpm) => (rpm >= 10500 ? PU.fuelCap : 0.009 * rpm + 5.5);
const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
export const hp = (kw) => kw * 1.34102;

export function makeState() {
  return { rpm: 4000, turbo: 30000, socMJ: 2, deployed: 0, harvested: 0, out: null };
}

/** Power balance for one instant (no memory except the spool speed passed in). */
export function balance(inp, turboRpm) {
  const thr = clamp(inp.throttle) * (1 - clamp(inp.brake));
  const rpm = inp.rpm;
  const flow = maxFlow(rpm) * thr; // kg/h
  const fuelKW = ((flow / 3600) * PU.lhv) / 1000;
  const eta = PU.etaPeak - 0.09 * (1 - thr);
  const iceKW = thr > 0.001 ? fuelKW * eta : 0;
  const omega = (rpm * 2 * Math.PI) / 60;
  const torque = iceKW > 0 ? (iceKW * 1000) / omega : 0; // N m
  // boost the turbine speed can support, and what the compressor then needs
  const spoolShare = clamp((turboRpm - 30000) / (PU.turboMax - 30000));
  const pr = 1 + 2.65 * spoolShare; // absolute pressure ratio
  const air = (flow / 3600) * 14.7 * 1.25; // kg/s
  const compKW = (air * 1005 * 300 * (Math.pow(pr, 0.2857) - 1)) / 0.76 / 1000;
  // turbine power rises with the exhaust energy and with how hard the turbo is already spinning
  const turbKW = PU.exhaustShare * fuelKW * (0.55 + 0.45 * Math.sqrt(spoolShare)) * (thr > 0.001 ? 1 : 0);
  const hKW = turbKW - compKW; // + = MGU-H generating, - = MGU-H motoring the shaft
  // MGU-K
  const deployLeft = Math.max(0, PU.deployPerLap - inp.deployed);
  const harvestLeft = Math.max(0, PU.harvestPerLap - inp.harvested);
  const socOk = inp.socMJ > 0.02;
  const full = inp.socMJ >= PU.storeWindow - 0.02;
  let k = 0;
  if (inp.mode === "deploy") k = PU.kMax * clamp((thr - 0.15) / 0.5);
  if (inp.mode === "balanced") k = 0.5 * PU.kMax * clamp((thr - 0.15) / 0.5);
  if (inp.mode === "harvest") k = -0.3 * PU.kMax * clamp(thr * 1.4);
  if (k > 0 && !(deployLeft > 0 && socOk)) k = 0;
  let brakeK = -PU.kMax * clamp(inp.brake);
  let kW = k + brakeK;
  if (kW < 0 && !(harvestLeft > 0 && !full)) kW = 0;
  // electrical balance: the store makes up what the MGU-H does not cover, or takes what is left
  const kElec = kW > 0 ? kW / 0.95 : kW * 0.95;
  const esKW = kElec - hKW; // + = store discharging
  const totalKW = iceKW + kW;
  return { thr, flow, fuelKW, iceKW, torque, pr, boost: pr, air, compKW, turbKW, hKW, kW, esKW, totalKW, totalHp: hp(totalKW), deployLeft, harvestLeft, spoolShare };
}

/** Advance rpm, turbo speed, store and lap counters by dt animation seconds. */
export function step(state, inp, dt) {
  const calmRpm = state.rpm + (inp.rpm - state.rpm) * (1 - Math.exp(-dt / 0.25));
  state.rpm = Math.abs(inp.rpm - calmRpm) < 1 ? inp.rpm : calmRpm;
  let b = balance({ ...inp, rpm: state.rpm, socMJ: state.socMJ, deployed: state.deployed, harvested: state.harvested }, state.turbo);
  // the turbo spools towards the speed the exhaust can hold; the MGU-H adds to or takes from the shaft
  const target = b.thr > 0.001 ? 30000 + (PU.turboMax - 30000) * clamp(Math.pow(b.thr, 0.8) * clamp(state.rpm / 11000)) : 30000;
  state.turbo += (target - state.turbo) * (1 - Math.exp(-dt / 0.55));
  b = balance({ ...inp, rpm: state.rpm, socMJ: state.socMJ, deployed: state.deployed, harvested: state.harvested }, state.turbo);
  const energyDt = dt * PU.timeLapse;
  state.socMJ = clamp(state.socMJ - (b.esKW * energyDt) / 1000, 0, PU.storeWindow);
  if (b.kW > 0) state.deployed += (b.kW * energyDt) / 1000;
  if (b.kW < 0) state.harvested += (-b.kW * energyDt) / 1000;
  state.out = b;
  return b;
}

export const newLap = (state) => { state.deployed = 0; state.harvested = 0; };

/* ---------- engine kinematics ---------- */

export const ENGINE = { crank: 13.25, rod: 70, pistonH: 24 }; // drawing units of 2 mm: stroke 53 mm, rod 140 mm

/** Distance from the crank axis to the piston pin along the cylinder axis. phi: crank angle from the cylinder axis. */
export const pinDistance = (phi) => {
  const { crank: r, rod: l } = ENGINE;
  return r * Math.cos(phi) + Math.sqrt(l * l - r * r * Math.sin(phi) * Math.sin(phi));
};

/** Firing angles in degrees of the 720 degree cycle, for the six cylinders, in this crank layout (uneven 90 / 150). */
export const FIRE = { A0: 45, A2: 165, A1: 285, B2: 435, B1: 555, B0: 675 };
export const BURN_DEG = 75;

/** 0..1 combustion glow for a cylinder at a crank angle (degrees, any size). */
export function burn(cycleDeg, fireDeg) {
  const a = (((cycleDeg - fireDeg) % 720) + 720) % 720;
  if (a > BURN_DEG) return 0;
  const t = a / BURN_DEG;
  return Math.min(1, t * 9) * Math.exp(-t * 3.2);
}

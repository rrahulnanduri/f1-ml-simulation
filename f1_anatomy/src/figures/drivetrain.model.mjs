// Gearbox, engine and steering-wheel settings model. Pure functions, SI units unless a name says otherwise.
// Rounded, typical modern Formula 1 values; the behaviour is the point, not a team's numbers.

export const WHEEL_RADIUS = 0.33; // m, rolling radius of the driven tyre
export const MASS = 800; // kg
export const CDA = 1.15; // m2, drag area
export const RHO = 1.2;
export const REV_LIMIT = 15000; // rpm
export const FINAL_DRIVE = 4.6; // crown wheel and pinion, so the pair ratio is the overall ratio over this

// Overall ratios from crank to wheel: first gear tops out near 85 km/h, eighth near 350 km/h at the limiter.
export const RATIOS = (() => {
  const first = ((REV_LIMIT * 2 * Math.PI) / 60 * WHEEL_RADIUS) / (85 / 3.6);
  const last = ((REV_LIMIT * 2 * Math.PI) / 60 * WHEEL_RADIUS) / (350 / 3.6);
  const f = Math.pow(last / first, 1 / 7);
  return Array.from({ length: 8 }, (_, i) => first * Math.pow(f, i));
})();

/** Gear-pair ratio of one gear (what the visible gears do), the rest is the crown wheel and pinion. */
export const pairRatio = (gear) => RATIOS[gear - 1] / FINAL_DRIVE;

export const SHIFT_MS = 28; // seamless-shift time, rounded

export const BRAKE_BALANCE = [54, 55, 56, 57, 58, 59, 60]; // % to the front, the driver's rotary
export const MODES = [
  { id: "save", label: "Save", power: 0.82, deploy: 0 },
  { id: "race", label: "Race", power: 0.94, deploy: 60 },
  { id: "quali", label: "Quali", power: 1.0, deploy: 120 },
];
export const DIFF = [
  { id: "open", label: "Open", lock: 0.1 },
  { id: "mid", label: "Mid", lock: 0.45 },
  { id: "locked", label: "Locked", lock: 0.8 },
];

/** Combustion engine power in kW at a crank speed, full throttle, for a mode. */
export function enginePower(rpm, mode = MODES[2]) {
  const x = (rpm - 11500) / 10000;
  const f = Math.max(0.2, 1 - x * x * 1.0);
  return 640 * f * mode.power;
}

export const omega = (rpm) => (rpm * 2 * Math.PI) / 60;

/** Road speed in m/s for a gear and engine speed. */
export const speedAt = (gear, rpm) => (omega(rpm) * WHEEL_RADIUS) / RATIOS[gear - 1];
export const rpmAt = (gear, v) => ((v * RATIOS[gear - 1]) / WHEEL_RADIUS) * (60 / (2 * Math.PI));

/** Traction force available at the wheel at throttle 0..1, N. */
export function driveForce(gear, rpm, throttle, mode) {
  const p = (enginePower(rpm, mode) * throttle + (throttle > 0.5 ? mode.deploy : 0) * Math.min(1, throttle)) * 1000;
  const w = omega(Math.max(rpm, 3000));
  const torque = p / w; // N m at the crank
  return (torque * RATIOS[gear - 1]) / WHEEL_RADIUS;
}

export const resistance = (v) => 0.5 * RHO * CDA * v * v + 60;

/** Crank torque in N m. */
export function crankTorque(gear, rpm, throttle, mode) {
  const p = (enginePower(rpm, mode) * throttle + (throttle > 0.5 ? mode.deploy : 0) * Math.min(1, throttle)) * 1000;
  return p / omega(Math.max(rpm, 3000));
}

/**
 * Advance the car one step. state: { v (m/s), gear (1..8), rpm, limiter (0..1), shiftAt }.
 * `now` is the clock in seconds; a shift cuts drive for SHIFT_MS.
 */
export function step(state, throttle, brake, mode, dt, now) {
  const inShift = now - state.shiftAt < SHIFT_MS / 1000;
  state.rpm = rpmAt(state.gear, state.v);
  let rpmNow = Math.max(1500, state.rpm);
  let force = 0;
  state.limiter = 0;
  if (!inShift) {
    if (rpmNow >= REV_LIMIT) {
      state.limiter = 1;
      rpmNow = REV_LIMIT;
    } else {
      force = driveForce(state.gear, rpmNow, throttle, mode);
      // the rear tyres can only pass on so much: weight on them plus the downforce that presses them down
      const grip = 1.8 * (0.54 * MASS * 9.81 + 0.6 * 0.5 * RHO * 4.0 * state.v * state.v);
      state.traction = Math.min(1, force / grip);
      force = Math.min(force, grip);
    }
  }
  const brakeForce = brake * 5.2 * 9.81 * MASS * Math.min(1, 0.4 + state.v / 30);
  const a = (force - resistance(state.v) - (state.v > 0.1 ? brakeForce : 0)) / MASS;
  state.v = Math.max(0, state.v + a * dt);
  state.rpm = Math.max(1500, rpmAt(state.gear, state.v));
  state.accel = a;
  return state;
}

export function shift(state, direction, now) {
  const to = Math.max(1, Math.min(8, state.gear + direction));
  if (to === state.gear) return false;
  state.gear = to;
  state.shiftAt = now;
  return true;
}

/**
 * Where each dog ring sits, -1 (toward the lower gear of its pair), 0 (free), +1 (toward the
 * higher gear). Ring k serves gears 2k+1 and 2k+2.
 */
export function ringsFor(gear) {
  const rings = [0, 0, 0, 0];
  if (gear >= 1) {
    const k = Math.floor((gear - 1) / 2);
    rings[k] = gear % 2 === 1 ? -1 : 1;
  }
  return rings;
}

/**
 * Fork track on the shift drum: displacement (-1..1) of fork k as a function of the drum's
 * position in gears (0..8, continuous). Fork k is at -1 in gear 2k+1, +1 in 2k+2, and eases
 * to 0 while any other gear is selected (a smooth, symmetric groove).
 */
export function trackAt(k, drumGear) {
  const lo = 2 * k + 1;
  const hi = 2 * k + 2;
  const d = drumGear;
  const smooth = (t) => t * t * (3 - 2 * t);
  const near = (g) => Math.max(0, 1 - Math.abs(d - g));
  return -smooth(near(lo)) + smooth(near(hi));
}

/** Lap-speed style helper for the readout: seconds to go from standstill to v, flat out, for the intro. */
export const kmh = (v) => v * 3.6;

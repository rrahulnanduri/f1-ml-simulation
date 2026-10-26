// Aerodynamic model of a Formula 1 car in a wind tunnel. Pure functions, SI units unless a name says otherwise.
// Simplified and rounded on purpose: it shows how each part contributes and how the settings trade
// downforce against drag. It is not any team's data.

export const AIR = { rho: 1.2 };

// areas times coefficients (m2) at the reference setting: ride height 45 mm, front flap 15 deg, rear wing 22 deg
export const BASE = {
  frontWing: 0.95,
  floor: 1.55,
  diffuser: 0.45,
  rearWing: 0.85,
  body: 0.2,
  dragFront: 0.16,
  dragFloor: 0.1,
  dragRear: 0.3,
  dragBody: 0.5, // wheels, bodywork, cooling
};

export const STALL_MM = 24; // below this the floor loses its seal and stalls
export const OPTIMUM_MM = 32; // best seal

export const kmh = (v) => v / 3.6;
export const toKmh = (v) => v * 3.6;

/** How well the floor seals against the road, 0 to 1.1, from the height of the plank above the road in mm. */
export function floorSeal(mm) {
  const high = mm >= OPTIMUM_MM ? 1 - 0.0125 * (mm - OPTIMUM_MM) : 1 + 0.006 * (OPTIMUM_MM - mm);
  const sealed = 1 / (1 + Math.exp(-(mm - STALL_MM) / 1.1)); // 0 below the stall height, 1 above
  return Math.max(0.05, high * (0.42 + 0.58 * sealed));
}

/** Coefficient areas (m2) for a setting. drs is 0 (closed) to 1 (fully open). */
export function areas({ ride, fw, rw, drs }) {
  const seal = floorSeal(ride);
  const stall = Math.max(0, fw - 26) * 0.045; // a front wing pushed past about 26 deg starts to stall
  const frontWing = Math.max(0.2, BASE.frontWing * (1 + 0.026 * (fw - 15)) - stall);
  const floor = BASE.floor * seal;
  const diffuser = BASE.diffuser * Math.pow(seal, 1.3);
  const rearAngle = 1 + 0.03 * (rw - 22);
  const rearWing = BASE.rearWing * rearAngle * (1 - 0.3 * drs);
  const body = BASE.body;
  const dragFront = BASE.dragFront * (1 + 0.032 * (fw - 15));
  const dragFloor = BASE.dragFloor * (0.7 + 0.3 * seal);
  const dragRear = BASE.dragRear * (1 + 0.052 * (rw - 22)) * (1 - 0.82 * drs);
  const dragBody = BASE.dragBody;
  return {
    cl: { frontWing, floor, diffuser, rearWing, body },
    cd: { dragFront, dragFloor, dragRear, dragBody },
    clTotal: frontWing + floor + diffuser + rearWing + body,
    cdTotal: dragFront + dragFloor + dragRear + dragBody,
    seal,
  };
}

/** Forces (N) at speed v (m/s). */
export function forces(setting, v) {
  const a = areas(setting);
  const q = 0.5 * AIR.rho * v * v;
  const part = {
    frontWing: q * a.cl.frontWing,
    floor: q * a.cl.floor,
    diffuser: q * a.cl.diffuser,
    rearWing: q * a.cl.rearWing,
    body: q * a.cl.body,
  };
  const down = part.frontWing + part.floor + part.diffuser + part.rearWing + part.body;
  // the floor works along the whole car, a little more at the front; the diffuser acts at the back
  const front = part.frontWing + 0.42 * part.floor + 0.35 * part.body;
  const rear = down - front;
  const drag = q * a.cdTotal;
  return { ...part, down, front, rear, drag, ld: drag > 0 ? down / drag : 0, floorShare: down > 0 ? (part.floor + part.diffuser) / down : 0, seal: a.seal };
}

/** Speed where engine power at the wheels balances drag plus rolling resistance (m/s). */
export function topSpeed(setting, power = 600e3) {
  const a = areas(setting);
  const need = (v) => {
    const q = 0.5 * AIR.rho * v * v;
    return (q * a.cdTotal + 0.012 * (800 * 9.81 + q * a.clTotal)) * v;
  };
  let lo = 20;
  let hi = 160;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (need(mid) < power) lo = mid;
    else hi = mid;
  }
  return lo;
}

/** The floor "porpoises" when it hovers around the stall height: loses the seal, rises, regains it, drops again. */
export const porpoising = (mm) => mm >= STALL_MM - 2 && mm <= STALL_MM + 4;

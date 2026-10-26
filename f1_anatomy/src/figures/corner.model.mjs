// Quarter-car model of one front corner: tyre, suspension, brake disc.
// Pure functions and constants, no DOM. SI units unless a name says otherwise.
// Figures are rounded, typical modern Formula 1 values; the point is the behaviour, not a team's numbers.

export const CAR = {
  mass: 800, // kg, car and driver
  front: 0.46, // static weight fraction on the front axle
  wheelbase: 3.6, // m
  cg: 0.28, // m, centre of gravity height
  track: 1.6, // m
  downforceArea: 4.0, // m2, Cl x A at full downforce
  frontDownforce: 0.4, // share of downforce on the front axle
  brakeBias: 0.58, // share of braking force on the front axle
  rho: 1.2, // kg/m3
  g: 9.81,
};

export const CORNER = {
  sprung: 158, // kg of the car resting on this corner
  unsprung: 28, // kg: wheel, tyre, upright, brake, half the arms
  wheelRate: 1.3e5, // N/m, spring plus heave element seen at the wheel
  bumpDamp: 5000, // N s/m, damper in compression at the wheel
  reboundDamp: 8000, // N s/m, damper in extension at the wheel
  tyreRate: 4.0e5, // N/m, vertical stiffness of the tyre
  tyreDamp: 250, // N s/m
  bumpStopAt: 0.045, // m of compression where the packers touch
  bumpStopRate: 2.5e6, // N/m
  reboundStopAt: -0.04,
  rollingRadius: 0.33, // m, unloaded tyre radius
};

export const DISC = {
  heat: 1400, // J/K, disc plus pad and caliper that share its heat
  radius: 0.139, // m
  air: 308, // K, ambient 35 C
  radiate: 7.6e-9, // W/K4, emissivity times area times Stefan-Boltzmann
  ductBase: 40, // W/K, convective cooling when standing
  ductSpeed: 1.6, // W/K per m/s of airspeed through the duct
};

export const kmh = (v) => v / 3.6;

/** Aerodynamic downforce on the whole car, N. */
export const downforce = (v) => 0.5 * CAR.rho * v * v * CAR.downforceArea;

/** Tyre friction coefficient falls as the load grows (load sensitivity). */
export const mu = (loadN) => 2.15 - 0.06 * (loadN / 1000);

/**
 * What the corner is asked to do and what it carries.
 * pedal 0..1, lat in g (this corner is the outside front wheel), v in m/s.
 * The tyre can only give mu x load in total, so when the request is larger than that the
 * achieved braking and cornering shrink together (the tyre is "at the limit").
 */
export function loads(pedal, lat, v) {
  const m = CAR.mass;
  const staticLoad = (CAR.front * m * CAR.g) / 2;
  const aero = 0.5 * CAR.frontDownforce * downforce(v);
  let decel = pedal * 5.5 * CAR.g;
  let ay = lat * CAR.g;
  let scale = 1;
  let load = staticLoad + aero;
  let fx = 0;
  let fy = 0;
  for (let i = 0; i < 6; i++) {
    const transferLong = (0.5 * m * decel * CAR.cg) / CAR.wheelbase;
    const transferLat = (CAR.front * m * ay * CAR.cg) / CAR.track;
    load = staticLoad + aero + transferLong + transferLat;
    fx = (CAR.brakeBias * m * decel) / 2;
    // the front axle's lateral force is shared between its two tyres in proportion to their loads
    fy = (CAR.front * m * ay * load) / (2 * (staticLoad + aero + transferLong));
    const demand = Math.hypot(fx, fy);
    const limit = mu(load) * load;
    scale = demand > limit ? limit / demand : 1;
    decel = pedal * 5.5 * CAR.g * scale;
    ay = lat * CAR.g * scale;
  }
  const limit = mu(load) * load;
  const demand = Math.hypot(fx, fy);
  return { staticLoad, aero, load, fx, fy, limit, usage: demand / limit, scale, decel, ay, mu: mu(load) };
}

/** Disc temperature rate, K/s. */
export function discRate(T, brakingW, v) {
  const air = DISC.ductBase + DISC.ductSpeed * v;
  const convect = air * (T - DISC.air);
  const radiate = DISC.radiate * (T ** 4 - DISC.air ** 4);
  return (brakingW - convect - radiate) / DISC.heat;
}

/** Braking power into this front disc, W. */
export const brakingPower = (fx, v) => fx * v;

/** Visible heat of a disc, 0 (black) to 1 (yellow-white), from temperature in K. */
export const glow = (T) => Math.min(1, Math.max(0, (T - 673) / (1373 - 673)));

/**
 * Quarter-car state: heights above the rig in metres.
 * xs sprung mass, xu wheel centre, with velocities. zr is the pad (road) height.
 */
export function makeCar() {
  return { xs: 0.3, vs: 0, xu: 0.3, vu: 0, restLength: 0.18, tyreDeflection: 0, force: 0, damper: 0, travel: 0, load: 0 };
}

/**
 * Advance the suspension by dt seconds. `extra` is the load on the corner beyond the weight of
 * the car that rests on it (downforce, braking and cornering transfer), in N.
 */
export function step(car, zr, extraN, dt) {
  const C = CORNER;
  const n = Math.max(1, Math.ceil(dt / 0.0004));
  const h = dt / n;
  for (let i = 0; i < n; i++) {
    const travel = car.xs - car.xu - car.restLength; // positive = extension
    const compression = -travel;
    const rate = car.vs - car.vu;
    const spring = C.wheelRate * compression;
    const damp = -(rate >= 0 ? C.reboundDamp : C.bumpDamp) * rate;
    let stop = 0;
    if (compression > C.bumpStopAt) stop = C.bumpStopRate * (compression - C.bumpStopAt);
    if (compression < C.reboundStopAt) stop = -C.bumpStopRate * (C.reboundStopAt - compression);
    const upOnSprung = spring + damp + stop;
    const deflect = C.rollingRadius - (car.xu - zr);
    let tyre = 0;
    if (deflect > 0) tyre = C.tyreRate * deflect + C.tyreDamp * Math.max(0, -car.vu);
    car.vs += ((upOnSprung - C.sprung * CAR.g - extraN) / C.sprung) * h;
    car.vu += ((tyre - upOnSprung - C.unsprung * CAR.g) / C.unsprung) * h;
    car.xs += car.vs * h;
    car.xu += car.vu * h;
    car.tyreDeflection = Math.max(0, deflect);
    car.force = tyre;
    car.damper = damp;
    car.travel = compression;
  }
  car.load = car.force;
  return car;
}

/** Settle the car on a flat pad under a steady extra load, for the start of the page. */
export function settleCar(car, extraN, seconds = 1.5) {
  const C = CORNER;
  car.xs = C.rollingRadius + car.restLength + 0.02;
  car.xu = C.rollingRadius - 0.01;
  for (let t = 0; t < seconds; t += 0.01) step(car, 0, extraN, 0.01);
  return car;
}

/** Kerb strike profile: height in metres at time t after the strike (ramp up, dwell, ramp down). */
export function kerb(t, height) {
  if (t < 0) return 0;
  if (t < 0.045) return height * (t / 0.045);
  if (t < 0.2) return height;
  if (t < 0.26) return height * (1 - (t - 0.2) / 0.06);
  return 0;
}

/** Contact patch length, m, from load and tyre width: the tyre flattens until area x pressure = load. */
export function patchLength(loadN) {
  const pressure = 1.4e5; // Pa, about 1.4 bar of effective contact pressure
  const width = 0.305;
  return Math.min(0.32, loadN / pressure / width);
}

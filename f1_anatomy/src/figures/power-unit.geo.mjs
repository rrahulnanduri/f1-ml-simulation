// Geometry of the power unit on its dyno bench: 90 degree V6, turbo with MGU-H, MGU-K, store.
// World units are 2 mm. x runs from the front of the engine (MGU-K) to the back (turbo), y across, z up.
// Runs in Node (the build draws the page) and in the browser (live parts are rebuilt each frame)
// through the namespaces k (iso kit), G (lathe) and T (tubes).
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as G from "../../vendor/kit/lathe.mjs";
import * as T from "../../vendor/kit/tube.mjs";

export const WIDTH = 960;
export const HEIGHT = 600;

const Z0 = 64; // crankshaft axis height
const ZT = Z0 + 72; // turbo shaft height
const S45 = Math.SQRT1_2;
const BORE_R = 20;
const CRANK_R = 13.25;
const ROD_L = 70;
const PITCH = 46;
const X_A = [20, 66, 112]; // bank +y cylinders
const X_B = [34, 80, 126]; // bank -y cylinders
const PIN_X = (i) => 27 + PITCH * i;
export const FIRE = { A0: 45, A2: 165, A1: 285, B2: 435, B1: 555, B0: 675 };
const BURN_DEG = 75;

export const BASE = { x: -170, y: -190, w: 640, d: 380, r: 12 };
const BASE_H = 12;
const FOOT_H = 6;

const dirOf = (sg) => [0, sg * S45, S45];
const outOf = (sg) => [0, sg * S45, -S45];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, f) => [a[0] * f, a[1] * f, a[2] * f];
const bankFrame = (sg, x) => G.frameOf([x, 0, Z0], dirOf(sg), [1, 0, 0], outOf(sg));
const axisX = (o) => G.frameOf(o, [1, 0, 0], [0, 1, 0], [0, 0, 1]);
const axisY = (o) => G.frameOf(o, [0, 1, 0], [1, 0, 0], [0, 0, 1]);
const axisZ = (o) => G.frameOf(o, [0, 0, 1], [1, 0, 0], [0, 1, 0]);

/** Distance from the crank axis to the piston pin along the cylinder axis. */
const pinDistance = (phi) => CRANK_R * Math.cos(phi) + Math.sqrt(ROD_L * ROD_L - CRANK_R * CRANK_R * Math.sin(phi) * Math.sin(phi));

export function burnOf(cycleDeg, fireDeg) {
  const a = (((cycleDeg - fireDeg) % 720) + 720) % 720;
  if (a > BURN_DEG) return 0;
  const t = a / BURN_DEG;
  return Math.min(1, t * 9) * Math.exp(-t * 3.2);
}

export function projection() {
  const pts = [...k.boxCorners(BASE, -BASE_H - FOOT_H, 0), [-90, 0, ZT + 90], [400, 0, ZT + 50], [150, 0, Z0 + 150]];
  return k.fitProjection(pts, WIDTH, HEIGHT, { pad: 22, azimuth: 124, elevation: 34 });
}

export const REST_POSE = () => poseOf({ crank: 30, turbo: 0, load: 0, exh: 0, air: 0, k: 0, h: 0, es: 0, flow: 0 });

/** What the drawing needs: crank angle in degrees (mod 720), turbo angle, flame strength and flow strengths. */
export function poseOf({ crank = 0, turbo = 0, load = 0, exh = 0, air = 0, k: kk = 0, h = 0, es = 0, flow = 0, soc = 0.5 }) {
  return { crank: ((crank % 720) + 720) % 720, turbo, load, exh, air, k: kk, h, es, flow, soc };
}

export function toolkit(P) {
  const S = (paths, style) => k.solidSvg(paths, style);
  const L = (d, style) => k.lineSvg(d, style);
  const D = (points, style) => k.dotsSvg(points, style);
  const at = (point) => k.iso(point, P);
  const view = G.viewOf(P);
  const depth = (p) => G.dot3(p, view);
  return { S, L, D, at, depth };
}

const polyPath = (pts, P) => k.pathOf(pts.map((p) => k.iso(p, P)));

function bar(a, b, r, P, options = {}) {
  const length = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const dir = [(b[0] - a[0]) / length, (b[1] - a[1]) / length, (b[2] - a[2]) / length];
  return G.disc(0, length, r, G.frameAlong(a, dir), P, options);
}

/* exhaust and intake routes, shared by the solid pipes and the flow lines drawn on them */
const OUT_PORT = (sg, x) => add([x, 0, Z0], add(mul(dirOf(sg), 118), mul(outOf(sg), 22)));
const MANI = (sg, x) => add(OUT_PORT(sg, x), mul(outOf(sg), 20));
function exhaustRoute(sg, P) {
  const m0 = MANI(sg, 0);
  const pts = [[14, m0[1], m0[2]], [204, m0[1], m0[2]], [236, m0[1], ZT], [356, m0[1], ZT], [356, sg * 52, ZT]];
  return G.fillet(pts, 22, 10, { P, tube: 8 });
}
const INTAKE_ROUTE = (P) => G.fillet([[250, 0, Z0 + 129], [232, 0, Z0 + 129], [208, 0, Z0 + 114], [160, 0, Z0 + 114]], 12, 8, { P, tube: 7 });
const COMPRESSOR_X = 246;

/* ---------- items ---------- */

export function items(P) {
  const { S, L, D, at, depth } = toolkit(P);
  const list = [];
  const put = (item) => list.push(item);
  const V = G.viewOf(P);

  const tubeMarkup = (route, r, tone, opts = {}) => {
    const pieces = T.tubePieces(route, r, P, { maxLength: opts.maxLength || 40, spacing: 1.6, ...opts.tube });
    return pieces
      .map((p) => ({ p, key: G.dot3(p.mid, V) }))
      .sort((a, b) => a.key - b.key)
      .map(({ p }) => T.tubeSvg(p, { tone }))
      .join("");
  };

  const flowStyle = (mag, speed) => `style="opacity:${(0.25 + 0.75 * Math.min(1, mag)).toFixed(2)};animation-duration:${(2.2 / (0.25 + speed)).toFixed(2)}s;stroke-width:1.4"`;
  const flowLine = (points, tone, mag, speed, reverse = false) => {
    if (mag < 0.03) return "";
    const pts = reverse ? [...points].reverse() : points;
    const d = polyPath(pts, P);
    return k.lineSvg(d, { tone, flow: true, free: true }).replace('class="iso-line"', `class="iso-line" ${flowStyle(mag, speed)}`);
  };

  /* --- bench --- */
  put({
    id: "base",
    group: "rig",
    svg: (() => {
      const out = [`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT_H, BASE_H, P)}"/>`];
      const sorted = [...k.corners(BASE, 26)].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
      for (const [x, y] of sorted) {
        out.push(S(k.cylinder(x, y, 9, -BASE_H - FOOT_H, 2, P, 28), { tone: "lo" }));
        out.push(S(k.cylinder(x, y, 7, -BASE_H - FOOT_H + 2, FOOT_H - 2, P, 28), { tone: "mid" }));
      }
      out.push(S(k.slabOf(BASE, -BASE_H, BASE_H, P, 8, 1.6), { tone: "mid" }));
      out.push(L(k.planOutline(k.insetPlan(BASE, 6), 0, P), { tone: "faint" }));
      const screws = k.corners(BASE, 12);
      out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.6, 0, P, 16)).join(""), { tone: "lo" }));
      out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
      const strip = { x: 30, y: BASE.y + BASE.d - 22, w: 320, d: 9, r: 1 };
      out.push(S(k.slabOf(strip, 0, 0.7, P, 3), { tone: "lo", crease: "none" }));
      const ticks = k.topTicks(strip.x + 5, strip.x + strip.w - 5, 5, 5, strip.y + 1, 0.7, [2.6, 4.4], P);
      out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
      out.push(D([at([strip.x + 2.5, strip.y + 4.5, 0.7]), at([strip.x + strip.w - 2.5, strip.y + 4.5, 0.7])], { size: 0.45 }));
      const label = { x: 250, y: BASE.y + 20, w: 90, d: 26, r: 2 };
      out.push(S(k.slabOf(label, 0, 0.9, P, 4), { tone: "lo", crease: "none" }));
      out.push(L(k.planOutline(k.insetPlan(label, 2.4), 0.9, P), { tone: "faint" }));
      [[8, 20, 70], [8, 13, 46], [8, 6, 58]].forEach(([dx, dy, len]) => out.push(L(k.lineOnTop([label.x + dx, label.y + dy], [label.x + dx + len, label.y + dy], 0.9, P), { tone: "lo", free: true })));
      out.push(D([at([label.x + 5, label.y + 5, 0.9]), at([label.x + label.w - 5, label.y + label.d - 5, 0.9])], { size: 0.45 }));
      return out.join("");
    })(),
  });

  /* --- dyno at the back of the shaft --- */
  put({
    id: "dyno",
    group: "rig",
    svg: (() => {
      const out = [];
      out.push(S(k.slabOf({ x: 262, y: -26, w: 34, d: 52, r: 3 }, 0, 40, P, 4, 0.6), { tone: "mid" }));
      const F = axisX([0, 0, Z0]);
      out.push(S(G.solidOf([[255, 31], [262, 33], [296, 33], [300, 30]], F, P, { smooth: false }), { tone: "mid", crease: "faint" }));
      out.push(L(G.arcOf(F, 270, 33, P) + G.arcOf(F, 288, 33, P), { tone: "faint" }));
      out.push(L(G.ribsOf(F, 262, 296, 33, 28, P, { seams: true }), { tone: "lo" }));
      out.push(S(G.solidOf([[300, 14], [310, 14], [310, 6], [322, 6]], F, P, { smooth: false }), { tone: "hi", crease: "lo" }));
      return out.join("");
    })(),
  });

  /* --- the far side: exhaust of bank -y --- */
  const exhaustSolid = (sg) => {
    const out = [];
    const route = exhaustRoute(sg, P);
    // log manifold along the heads, three stubs from the ports, flanges
    out.push(tubeMarkup(route, 8, "hi", { maxLength: 36 }));
    (sg > 0 ? X_A : X_B).forEach((x) => {
      const port = OUT_PORT(sg, x);
      const F = G.frameAlong(port, outOf(sg));
      out.push(S(G.disc(0, 13, 5.2, F, P, { steps: 18 }), { tone: "mid", crease: "none" }));
      out.push(S(G.disc(0, 3, 9, F, P, { steps: 24 }), { tone: "hi", crease: "lo" }));
    });
    const mp = MANI(sg, 0);
    out.push(S(G.disc(0, 4, 11, G.frameOf([14, mp[1], mp[2]], [1, 0, 0], [0, 1, 0], [0, 0, 1]), P, { steps: 24 }), { tone: "hi", crease: "lo" }));
    return out.join("");
  };
  const EXH_FAR = exhaustSolid(-1);
  put({ id: "exhaust-far", group: "exhaust", deps: ["exh"], live: (pose) => EXH_FAR + flowLine(exhaustRoute(-1, P), "red", pose.exh, pose.exh) });

  /* --- bank -y: studs, pistons, heads (far side, drawn first) --- */
  const studs = (sg, xs, back) => {
    const out = [];
    xs.forEach((x) => {
      const F = bankFrame(sg, x);
      const corners = [[-22, -20], [22, -20], [-22, 20], [22, 20]];
      corners
        .map(([u, v]) => ({ u, v, key: depth(add(add([x, 0, Z0], mul(dirOf(sg), 66)), add(mul([1, 0, 0], u), mul(outOf(sg), v)))) }))
        .sort((a, b) => a.key - b.key)
        .forEach((c, index) => {
          const isBack = index < 2;
          if (isBack !== back) return;
          const G2 = G.frameOf(add(add([x, 0, Z0], mul([1, 0, 0], c.u)), mul(outOf(sg), c.v)), dirOf(sg), [1, 0, 0], outOf(sg));
          out.push(S(G.disc(30, 104, 1.8, G2, P, { steps: 12 }), { tone: "mid", crease: "none" }));
        });
    });
    return out.join("");
  };
  const cylLive = (sg, xs, fires, pose) => {
    const out = [];
    xs.forEach((x, i) => {
      const F = bankFrame(sg, x);
      const alpha = ((pose.crank + 120 * i) * Math.PI) / 180; // pin angle from +z towards +y
      const phi = alpha - sg * (Math.PI / 4);
      const sp = pinDistance(phi);
      const pinPos = [x, CRANK_R * Math.sin(alpha), Z0 + CRANK_R * Math.cos(alpha)];
      const pistonPin = add([x, 0, Z0], mul(dirOf(sg), sp));
      // rod: big end on the crank pin, small end in the piston
      out.push(S(bar(pinPos, pistonPin, 2.8, P, { steps: 10 }), { tone: "hi", crease: "none" }));
      out.push(S(G.disc(-4, 4, 6.4, G.frameOf(pinPos, [1, 0, 0], [0, 1, 0], [0, 0, 1]), P, { steps: 16 }), { tone: "mid", crease: "lo" }));
      // piston
      out.push(S(G.disc(sp - 14, sp + 10, BORE_R - 1, F, P, { steps: 20 }), { tone: "hi", crease: "lo" }));
      out.push(L(G.arcOf(F, sp + 5, BORE_R - 1, P) + G.arcOf(F, sp + 1.5, BORE_R - 1, P), { tone: "faint" }));
      out.push(S(G.disc(-BORE_R + 4, BORE_R - 4, 3.3, G.frameOf(pistonPin, [1, 0, 0], [0, 1, 0], [0, 0, 1]), P, { steps: 12 }), { tone: "lo", crease: "none" }));
      // combustion, seen through the glass sleeve as a glow between the crown and the head
      const name = (sg > 0 ? "A" : "B") + (sg > 0 ? i : i);
      const b = burnOf(pose.crank, fires[i]) * pose.load;
      if (b > 0.03) {
        const glow = G.disc(sp + 10.5, 104, BORE_R - 2, F, P, { steps: 20 });
        out.push(`<path d="${glow.fill}" style="fill:var(--fa-accent);opacity:${(0.18 + 0.55 * b).toFixed(3)};stroke:none"/>`);
      }
      void name;
    });
    return out.join("");
  };
  const heads = (sg, xs, cut = false) => {
    const out = [];
    const F0 = bankFrame(sg, xs[0]);
    xs.forEach((x) => {
      const F = bankFrame(sg, x);
      out.push(S(G.prismOf(F, [[-22, -22], [22, -22], [22, 22], [-22, 22]], 104, cut ? 112 : 122, P), { tone: "mid", crease: "faint" }));
    });
    if (!cut) {
      out.push(S(G.prismOf(F0, [[-24, -15], [PITCH * 2 + 24, -15], [PITCH * 2 + 24, 15], [-24, 15]], 122, 136, P), { tone: "hi", crease: "lo" }));
      xs.forEach((x) => {
        const F = G.frameOf([x, 0, Z0], dirOf(sg), [1, 0, 0], outOf(sg));
        out.push(S(G.disc(136, 144, 3, F, P, { steps: 14 }), { tone: "lit", crease: "none", lit: true }));
      });
    }
    return out.join("");
  };
  const FIRES_B = [FIRE.B0, FIRE.B1, FIRE.B2];
  const FIRES_A = [FIRE.A0, FIRE.A1, FIRE.A2];

  /* --- turbocharger: compressor, MGU-H, turbine on one shaft --- */
  const FT = axisX([0, 0, ZT]);
  put({
    id: "turbo-base",
    group: "turbo",
    svg: (() => {
      const out = [];
      // support frame on the bench
      out.push(S(k.slabOf({ x: 252, y: -22, w: 22, d: 44, r: 3 }, 22, ZT - 34 - 22, P, 3, 0.5), { tone: "mid" }));
      out.push(S(k.slabOf({ x: 346, y: -22, w: 22, d: 44, r: 3 }, 22, ZT - 38 - 22, P, 3, 0.5), { tone: "mid" }));
      return out.join("");
    })(),
  });
  put({
    id: "compressor",
    group: "turbo",
    svg: (() => {
      const out = [];
      // inlet bell, then the volute
      out.push(S(G.solidOf([[222, 26], [230, 20], [238, 15, 1], [COMPRESSOR_X, 13]], FT, P, { smooth: true }), { tone: "mid", crease: "faint" }));
      out.push(S(G.solidOf([[COMPRESSOR_X, 14], [COMPRESSOR_X + 4, 26], [COMPRESSOR_X + 12, 34], [COMPRESSOR_X + 38, 34], [COMPRESSOR_X + 46, 28]], FT, P, { smooth: false }), { tone: "hi", crease: "lo" }));
      out.push(L(G.arcOf(FT, COMPRESSOR_X + 20, 34, P) + G.arcOf(FT, COMPRESSOR_X + 32, 34, P), { tone: "faint" }));
      out.push(L(G.ribsOf(FT, COMPRESSOR_X + 12, COMPRESSOR_X + 38, 34, 24, P, { seams: true }), { tone: "lo" }));
      // outlet pipe up to the intercooler
      out.push(S(G.disc(0, 14, 11, G.frameAlong([270, 0, ZT + 28], [0, 0, 1]), P, { steps: 22 }), { tone: "hi", crease: "lo" }));
      return out.join("");
    })(),
  });
  put({
    id: "mgu-h",
    group: "mguh",
    svg: (() => {
      const out = [];
      out.push(S(G.disc(292, 338, 24, FT, P, { steps: 36 }), { tone: "hi", crease: "lo" }));
      out.push(S(G.disc(292, 297, 28, FT, P, { steps: 36 }), { tone: "mid", crease: "faint" }));
      out.push(S(G.disc(333, 338, 28, FT, P, { steps: 36 }), { tone: "mid", crease: "faint" }));
      out.push(L(G.ribsOf(FT, 297, 333, 24, 32, P, { seams: true }), { tone: "lo" }));
      out.push(S(k.slabOf({ x: 306, y: 22, w: 20, d: 16, r: 2 }, ZT - 16, 18, P, 3, 0.4), { tone: "mid" }));
      out.push(D([at([312, 38, ZT - 8]), at([320, 38, ZT - 8])], { size: 0.6, tone: "hi" }));
      return out.join("");
    })(),
  });
  put({
    id: "turbine",
    group: "turbo",
    svg: (() => {
      const out = [];
      out.push(S(G.solidOf([[338, 28], [346, 36], [354, 40], [372, 40], [382, 30]], FT, P, { smooth: false }), { tone: "hi", crease: "lo" }));
      out.push(L(G.arcOf(FT, 360, 40, P) + G.arcOf(FT, 372, 40, P), { tone: "faint" }));
      // inlet bosses on both sides, where the exhaust pipes land
      [-1, 1].forEach((sg) => {
        const F = G.frameAlong([356, sg * 36, ZT], [0, sg, 0]);
        out.push(S(G.disc(0, 14, 12, F, P, { steps: 24 }), { tone: "mid", crease: "lo" }));
      });
      // exhaust outlet
      out.push(S(G.ringBand(382, 408, 15, 21, FT, P, { steps: 28 }), { tone: "mid", crease: "faint" }));
      out.push(`<path d="${G.circleOf(FT, 408, 15, P)}" style="fill:var(--anatomy-shade-0);stroke:none"/>`);
      return out.join("");
    })(),
  });
  put({
    id: "turbine-spin",
    group: "turbo",
    deps: ["turbo"],
    live: (pose) => {
      const phase = pose.turbo;
      const a = G.spokesOf(FT, 396, 3, 14, 11, P, { phase, sweep: 0.9 });
      return L(a, { tone: "hi", free: true }) + S(G.disc(396, 399, 3, FT, P, { steps: 14 }), { tone: "mid", crease: "none" });
    },
  });
  put({
    id: "turbo-spin",
    group: "mguh",
    deps: ["turbo"],
    live: (pose) => {
      const phase = pose.turbo;
      return D(G.dotsOf(FT, 338, 20, 8, P, { phase, all: true }), { size: 0.7, tone: "hi" }) + L(G.ribsOf(FT, 338, 342, 24, 16, P, { phase }), { tone: "lo", free: true });
    },
  });
  put({
    id: "intercooler",
    group: "intake",
    svg: (() => {
      const out = [];
      const plan = { x: 252, y: -38, w: 82, d: 76, r: 4 };
      out.push(S(k.slabOf(plan, Z0 + 112, 34, P, 4, 1), { tone: "mid" }));
      // core fins as a grid of grooves on the top, tank ends
      out.push(L(k.planOutline(k.insetPlan(plan, 12), Z0 + 146, P, 4), { tone: "faint" }));
      const t = k.topTicks(plan.x + 14, plan.x + plan.w - 14, 4, 4, plan.y + 12, Z0 + 146, [plan.d - 24, plan.d - 24], P);
      out.push(L(t.minor, { tone: "faint" }));
      out.push(L(k.sideSeam(plan, Z0 + 124, P, 4), { tone: "faint" }));
      out.push(D(k.corners(plan, 6).map(([x, y]) => at([x, y, Z0 + 146])), { size: 0.5 }));
      // outlet stub towards the plenum
      out.push(S(G.disc(0, 6, 9, G.frameOf([252, 0, Z0 + 129], [-1, 0, 0], [0, 1, 0], [0, 0, 1]), P, { steps: 20 }), { tone: "hi", crease: "lo" }));
      return out.join("");
    })(),
  });

  const INTAKE_PIPE = tubeMarkup(INTAKE_ROUTE(P), 7, "hi", { maxLength: 24 });
  put({ id: "intake-pipe", group: "intake", deps: ["air"], live: (pose) => INTAKE_PIPE + flowLine(INTAKE_ROUTE(P), "blue", pose.air, pose.air, true) });

  put({ id: "studs-far-back", group: "ice", svg: studs(-1, X_B, true) });
  put({ id: "cyl-far", group: "ice", deps: ["crank", "load"], live: (pose) => cylLive(-1, X_B, FIRES_B, pose) });
  put({ id: "studs-far-front", group: "ice", svg: studs(-1, X_B, false) });
  put({ id: "heads-far", group: "ice", svg: heads(-1, X_B) });

  /* --- crank bed, main bearings and the crankshaft --- */
  put({
    id: "bed",
    group: "ice",
    svg: (() => {
      const out = [];
      out.push(S(k.slabOf({ x: -16, y: -32, w: 188, d: 64, r: 4 }, 22, 12, P, 4, 0.8), { tone: "mid" }));
      out.push(L(k.planOutline(k.insetPlan({ x: -16, y: -32, w: 188, d: 64, r: 4 }, 5), 34, P, 4), { tone: "faint" }));
      [[-8, -26], [160, -26], [-8, 26], [160, 26]].forEach(([x, y]) => out.push(S(k.cylinder(x, y, 5, 0, 22, P, 20), { tone: "mid", crease: "none" })));
      out.push(D([[-6, -26], [-6, 26], [158, -26], [158, 26], [76, -26], [76, 26]].map(([x, y]) => at([x, y, 34])), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({
    id: "journal",
    group: "ice",
    svg: (() => {
      const F = axisX([0, 0, Z0]);
      return S(G.disc(-14, 198, 7.5, F, P, { steps: 24 }), { tone: "hi", crease: "none" }) + L(G.arcOf(F, -4, 7.5, P) + G.arcOf(F, 190, 7.5, P), { tone: "faint" });
    })(),
  });
  put({
    id: "mains",
    group: "ice",
    svg: [-4, 50, 96, 150]
      .map((c) => S(k.slabOf({ x: c - 5, y: -22, w: 10, d: 44, r: 2 }, 34, Z0 + 14 - 34, P, 3, 0.5), { tone: "mid" }) + L(k.lineOnTop([c - 5, 0], [c + 5, 0], Z0 + 14, P), { tone: "faint" }) + D([at([c, -16, Z0 + 14]), at([c, 16, Z0 + 14])], { size: 0.55, tone: "hi" }))
      .join(""),
  });
  const webPoints = (alpha) => {
    const pts = [];
    const circle = (cu, cv, r, n = 10) => {
      for (let a = 0; a < n; a++) pts.push([cu + r * Math.cos((a / n) * Math.PI * 2), cv + r * Math.sin((a / n) * Math.PI * 2)]);
    };
    circle(0, 0, 10);
    circle(CRANK_R * Math.sin(alpha), CRANK_R * Math.cos(alpha), 6.5, 8);
    for (let a = -58; a <= 58; a += 14) {
      const t = alpha + Math.PI + (a * Math.PI) / 180;
      pts.push([22 * Math.sin(t), 22 * Math.cos(t)]);
    }
    return G.hullOf(pts);
  };
  put({
    id: "crank",
    group: "ice",
    deps: ["crank"],
    live: (pose) => {
      const out = [];
      for (let i = 0; i < 3; i++) {
        const alpha = ((pose.crank + 120 * i) * Math.PI) / 180;
        const x0 = PIN_X(i) - 13;
        const x1 = PIN_X(i) + 13;
        const F = axisX([0, 0, Z0]);
        const pin = [PIN_X(i), CRANK_R * Math.sin(alpha), Z0 + CRANK_R * Math.cos(alpha)];
        const poly = webPoints(alpha);
        // viewer sits on the +x side: the web at +x is nearer
        out.push(S(G.prismOf(F, poly, x1 - 2, x1 + 3, P), { tone: "mid", crease: "lo" }));
        out.push(S(G.disc(x0 + 2, x1 - 2, 5.4, G.frameOf([0, pin[1], pin[2]], [1, 0, 0], [0, 1, 0], [0, 0, 1]), P, { steps: 20 }), { tone: "hi", crease: "none" }));
        out.push(S(G.prismOf(F, poly, x0 - 3, x0 + 2, P), { tone: "mid", crease: "lo" }));
      }
      return out.join("");
    },
  });

  /* --- front: MGU-K and its coupling; rear: flywheel, shaft --- */
  put({
    id: "flywheel",
    group: "ice",
    svg: (() => {
      const out = [];
      const F = axisX([0, 0, Z0]);
      out.push(S(G.disc(174, 190, 34, F, P, { steps: 56, bevel: 0.6 }), { tone: "hi", crease: "lo" }));
      out.push(S(G.disc(190, 196, 14, F, P, { steps: 28 }), { tone: "mid", crease: "faint" }));
      out.push(S(G.disc(196, 252, 8, F, P, { steps: 24 }), { tone: "mid", crease: "none" }));
      out.push(S(G.disc(244, 255, 14, F, P, { steps: 28 }), { tone: "hi", crease: "lo" }));
      out.push(L(G.circleOf(F, 190, 26, P), { tone: "faint" }));
      return out.join("");
    })(),
  });
  put({
    id: "flywheel-spin",
    group: "ice",
    deps: ["crank"],
    live: (pose) => {
      const F = axisX([0, 0, Z0]);
      const phase = (pose.crank * Math.PI) / 180;
      return L(G.ribsOf(F, 174, 190, 34, 60, P, { phase, seams: true }), { tone: "lo" }) + D(G.dotsOf(F, 190, 21, 8, P, { phase, all: true }), { size: 0.7, tone: "hi" }) + L(G.spokesOf(F, 190, 15, 19, 8, P, { phase }), { tone: "mid", free: true });
    },
  });

  /* --- intake: plenum in the valley, intercooler and pipe from the compressor --- */
  put({
    id: "plenum",
    group: "intake",
    svg: (() => {
      const out = [];
      const plan = { x: 16, y: -24, w: 146, d: 48, r: 7 };
      out.push(S(k.slabOf(plan, Z0 + 96, 24, P, 6, 1.2), { tone: "hi" }));
      out.push(L(k.sideSeam(plan, Z0 + 108, P, 6), { tone: "faint" }));
      out.push(L(k.planOutline(k.insetPlan(plan, 6), Z0 + 120, P, 6), { tone: "faint" }));
      // six velocity stacks on the lid: a ring each
      [...X_A, ...X_B].sort((a, b) => a - b).forEach((x, i) => {
        const y = i % 2 ? 8 : -8;
        out.push(S(k.cylinder(x + 4, y, 6, Z0 + 120, 5, P, 18), { tone: "mid" }));
        out.push(L(k.ring(x + 4, y, 3.4, Z0 + 125, P, 14), { tone: "lo" }));
      });
      return out.join("");
    })(),
  });
  /* --- the near side: bank +y, exhaust of bank +y --- */
  put({ id: "studs-near-back", group: "ice", svg: studs(1, X_A, true) });
  put({ id: "cyl-near", group: "ice", deps: ["crank", "load"], live: (pose) => cylLive(1, X_A, FIRES_A, pose) });
  put({ id: "studs-near-front", group: "ice", svg: studs(1, X_A, false) });
  put({ id: "heads-near", group: "ice", svg: heads(1, X_A, true) });
  const EXH_NEAR = exhaustSolid(1);
  put({ id: "exhaust-near", group: "exhaust", deps: ["exh"], live: (pose) => EXH_NEAR + flowLine(exhaustRoute(1, P), "red", pose.exh, pose.exh) });

  put({
    id: "mgu-k",
    group: "mguk",
    svg: (() => {
      const out = [];
      const F = axisX([0, 0, Z0]);
      out.push(S(k.slabOf({ x: -80, y: -26, w: 36, d: 52, r: 3 }, 22, 12, P, 3, 0.5), { tone: "mid" }));
      out.push(S(G.solidOf([[-86, 22], [-84, 26], [-46, 26], [-44, 23]], F, P, { smooth: false }), { tone: "hi", crease: "lo" }));
      out.push(L(G.ribsOf(F, -84, -46, 26, 30, P, { seams: true }), { tone: "lo" }));
      out.push(L(G.arcOf(F, -74, 26, P) + G.arcOf(F, -56, 26, P), { tone: "faint" }));
      // terminal box on the +y side
      out.push(S(k.slabOf({ x: -74, y: 22, w: 22, d: 18, r: 2 }, Z0 + 4, 14, P, 3, 0.4), { tone: "mid" }));
      out.push(D([at([-68, 38, Z0 + 18]), at([-58, 38, Z0 + 18])], { size: 0.6, tone: "hi" }));
      // coupling flange and gear case
      out.push(S(G.disc(-44, -36, 18, F, P, { steps: 32 }), { tone: "hi", crease: "lo" }));
      out.push(S(G.disc(-36, -14, 12, F, P, { steps: 28 }), { tone: "mid", crease: "faint" }));
      return out.join("");
    })(),
  });
  put({
    id: "mgu-k-spin",
    group: "mguk",
    deps: ["crank"],
    live: (pose) => {
      const F = axisX([0, 0, Z0]);
      const phase = (pose.crank * Math.PI) / 180;
      return D(G.dotsOf(F, -36, 14.5, 10, P, { phase, all: true }), { size: 0.7, tone: "hi" }) + L(G.spokesOf(F, -14, 4, 11, 6, P, { phase }), { tone: "lo", free: true });
    },
  });
  /* --- the electrics: energy store and control electronics on the bench --- */
  put({
    id: "store",
    group: "es",
    svg: (() => {
      const out = [];
      const plan = { x: -152, y: 78, w: 80, d: 72, r: 4 };
      out.push(S(k.slabOf(plan, 0, 36, P, 4, 0.9), { tone: "mid" }));
      out.push(L(k.sideSeam(plan, 18, P, 4), { tone: "faint" }));
      // cell modules on top
      [0, 1, 2].forEach((i) => out.push(S(k.slabOf({ x: plan.x + 6, y: plan.y + 6 + i * 21, w: 68, d: 18, r: 2 }, 36, 8, P, 3, 0.4), { tone: "hi", crease: "lo" })));
      out.push(D(k.corners(plan, 5).map(([x, y]) => at([x, y, 36])), { size: 0.5 }));
      // two terminals
      [[plan.x + 12, plan.y + 6], [plan.x + 24, plan.y + 6]].forEach(([x, y]) => out.push(S(k.cylinder(x, y, 3, 44, 5, P, 14), { tone: "lit", crease: "none", lit: true })));
      return out.join("");
    })(),
  });
  put({
    id: "store-level",
    group: "es",
    deps: ["soc"],
    live: (pose) => {
      // charge shown as a row of cell windows on the front face, filled from the left
      const plan = { x: -152, y: 78, w: 80, d: 72 };
      const out = [];
      const cells = 8;
      for (let i = 0; i < cells; i++) {
        const lit = (i + 0.5) / cells <= pose.soc;
        const x0 = plan.x + 6 + i * 9;
        const a = at([x0, plan.y + plan.d, 8]);
        const b = at([x0 + 6, plan.y + plan.d, 8]);
        const c = at([x0 + 6, plan.y + plan.d, 26]);
        const d = at([x0, plan.y + plan.d, 26]);
        out.push(`<path d="M${a[0]} ${a[1]}L${b[0]} ${b[1]}L${c[0]} ${c[1]}L${d[0]} ${d[1]}Z" style="fill:${lit ? "var(--anatomy-green)" : "var(--anatomy-faint)"};opacity:${lit ? 0.85 : 0.5};stroke:var(--anatomy-lo);stroke-width:.4;vector-effect:non-scaling-stroke"/>`);
      }
      return out.join("");
    },
  });
  put({
    id: "electronics",
    group: "ce",
    svg: (() => {
      const out = [];
      const plan = { x: -58, y: 84, w: 54, d: 58, r: 3 };
      out.push(S(k.slabOf(plan, 0, 30, P, 4, 0.8), { tone: "mid" }));
      const t = k.topTicks(plan.x + 6, plan.x + plan.w - 6, 4, 4, plan.y + 6, 30, [plan.d - 12, plan.d - 12], P);
      out.push(L(t.minor, { tone: "lo" }));
      out.push(L(k.planOutline(k.insetPlan(plan, 4), 30, P, 4), { tone: "faint" }));
      out.push(S(k.cylinder(plan.x + 12, plan.y + plan.d, 4, 8, 6, P, 16), { tone: "hi" }));
      out.push(S(k.cylinder(plan.x + 26, plan.y + plan.d, 4, 8, 6, P, 16), { tone: "hi" }));
      return out.join("");
    })(),
  });

  /* --- energy flows, drawn last over the hardware --- */
  const exhaustPts = (sg) => exhaustRoute(sg, P);
  const intakeA = [[222, 0, ZT], [260, 0, ZT], [270, 0, ZT + 28], [270, 0, Z0 + 112], [285, 0, Z0 + 129], [252, 0, Z0 + 129]];
  const intakeB = (() => INTAKE_ROUTE(P))();
  const KTERM = [-63, 38, Z0 + 18];
  const HTERM = [316, 37, ZT - 8];
  const STORE_TERM = [-128, 84, 44];
  const CE_TOP = [-31, 113, 30];
  put({
    id: "flows",
    group: "flows",
    deps: ["k", "h", "es"],
    live: (pose) => {
      const out = [];
      // electricity: store <-> electronics <-> K and H. + k means the K is driving, so power flows to it.
      const kMag = Math.abs(pose.k);
      const hMag = Math.abs(pose.h);
      const kTone = pose.k >= 0 ? "lit" : "green";
      const hTone = pose.h >= 0 ? "green" : "lit";
      const ceToK = [CE_TOP, [-31, 100, 46], [-48, 70, Z0 + 4], KTERM];
      const ceToH = [CE_TOP, [-31, 150, 10], [316, 150, 10], [316, 150, 118], [316, 36, 118], [316, 36, ZT - 6], HTERM];
      const esToCe = [STORE_TERM, [-100, 86, 54], [-48, 96, 44], CE_TOP];
      out.push(flowLine(ceToK, kTone, kMag, kMag, pose.k < 0));
      out.push(flowLine(ceToH, hTone, hMag, hMag, pose.h >= 0));
      const esMag = Math.abs(pose.es);
      out.push(flowLine(esToCe, pose.es >= 0 ? "lit" : "green", esMag, esMag, pose.es < 0));
      return out.join("");
    },
  });
  return list;
}

export const markup = (item, pose) => (item.live ? item.live(pose) : item.svg || "");

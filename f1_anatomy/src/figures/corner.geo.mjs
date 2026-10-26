// Geometry of the corner rig: one front wheel with its suspension, brake and tyre on a test rig.
// World units are 2 mm. x runs inboard (towards the car's centre line), y forward, z up.
// Everything here runs both in Node (the build draws the page) and in the browser (the live page
// redraws what moves), through the namespaces k (iso kit), G (lathe) and T (tubes).
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as G from "../../vendor/kit/lathe.mjs";

export const U = 0.002; // metres per unit
export const WIDTH = 960;
export const HEIGHT = 580;

export const PAD0 = 52; // pad top at rest
export const R_TYRE = 165; // 0.33 m
export const ZC0 = 215; // wheel centre at rest
export const ZS0 = ZC0 - 52; // chassis datum at rest (lower wishbone pick-up height)
const R_OPEN = 100; // radius of the open end of the rim
const OPEN_X = 82; // plane of the open end of the rim

export const BASE = { x: -160, y: -190, w: 620, d: 405, r: 12 };
const BASE_H = 12;
const FOOT_H = 6;
const PAD = { x: -104, y: -150, w: 208, d: 300, r: 9 };
const PAD_H = 10;
const ACT = [
  [0, -92],
  [0, 92],
];

const WX = 290; // the chassis stub's wheel-side face
const WALL = { x: WX, y: -84, w: 112, d: 168, r: 5 };
const WALL_BOTTOM = ZS0 - 40;
const WALL_TOP = ZS0 + 130;
const Y0 = 34; // the plane of the rocker, push rod and damper
const PIVOT = [WX + 30, Y0, WALL_TOP + 36]; // rocker pivot; the torsion bar runs along y through it
const ROCK_A = 54;
const ROCK_B = 48;
const ROCK_A0 = (200 * Math.PI) / 180;
const ROCK_B0 = (55 * Math.PI) / 180;
const DAMPER_MOUNT = [WX + 100, Y0, WALL_TOP + 48];
const ROD_X = WX + 126;
const ROD_Y = [-52, 52];
const BLOCK_H = 36;

const LOWER_IN = [
  [WX, 56, ZS0],
  [WX, -48, ZS0 + 4],
];
const UPPER_IN = [
  [WX, 50, ZS0 + 100],
  [WX, -44, ZS0 + 104],
];
const TRACK_IN = [WX, -62, ZS0 + 52];
const BALL_LOW = [98, 0, -64]; // relative to the wheel centre
const BALL_UP = [98, 0, 66];
const TRACK_OUT = [112, -34, 8];
const PUSH_OUT = [106, Y0, -50];

const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const PUSH_REST = (() => {
  const out = add([0, 0, ZC0], PUSH_OUT);
  const tip = [PIVOT[0] + ROCK_A * Math.cos(ROCK_A0), PIVOT[1], PIVOT[2] + ROCK_A * Math.sin(ROCK_A0)];
  return dist3(out, tip);
})();

export function projection() {
  const pts = [
    ...k.boxCorners(BASE, -BASE_H - FOOT_H, 0),
    [-90, 0, ZC0 + R_TYRE + 8],
    [ROD_X + 20, 0, WALL_TOP + 100],
    [0, 150, ZC0],
  ];
  return k.fitProjection(pts, WIDTH, HEIGHT, { pad: 26, azimuth: 52, elevation: 30 });
}

/* ---------- pose ---------- */

/** Everything the drawing needs, from the model state (heights in metres above the rig). */
export function poseOf({ xs, xu, zr, spin = 0, temp = 300, usage = 0, fxs = 0, fys = 0, patch = 0.04, glowAmount = 0, flow = 0, ghost = 0 }) {
  const zc = PAD0 + xu / U;
  const zs = zc - 52 + (xs - xu - 0.168) / U;
  const pose = {
    zr, zc, zs, spin, temp, usage, fxs, fys, patch, glowAmount, flow, ghost,
    padShift: zr / U,
    wheelShift: zc - ZC0,
    chassisShift: zs - ZS0,
  };
  pose.rocker = solveRocker(pose);
  return pose;
}

/** The rocker angle that keeps the push rod its true length, for the current wheel and chassis heights. */
function solveRocker(pose) {
  const out = [PUSH_OUT[0], pose.zc + PUSH_OUT[2]];
  const pivot = [PIVOT[0], PIVOT[2] + pose.chassisShift];
  const dx = out[0] - pivot[0];
  const dz = out[1] - pivot[1];
  const d = Math.hypot(dx, dz);
  const a = (ROCK_A * ROCK_A - PUSH_REST * PUSH_REST + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(0, ROCK_A * ROCK_A - a * a));
  const mx = pivot[0] + (a * dx) / d;
  const mz = pivot[1] + (a * dz) / d;
  const options = [
    Math.atan2(mz + (h * dx) / d - pivot[1], mx - (h * dz) / d - pivot[0]),
    Math.atan2(mz - (h * dx) / d - pivot[1], mx + (h * dz) / d - pivot[0]),
  ];
  const wrap = (v) => Math.abs(((v - ROCK_A0 + Math.PI * 3) % (Math.PI * 2)) - Math.PI);
  return wrap(options[0]) > wrap(options[1]) ? options[1] : options[0];
}

export const REST_POSE = () => poseOf({ xs: (ZC0 - PAD0) * U + 0.168, xu: (ZC0 - PAD0) * U, zr: 0, patch: 0.045 });

/* ---------- drawing helpers ---------- */

export function toolkit(P) {
  const S = (paths, style) => k.solidSvg(paths, style);
  const L = (d, style) => k.lineSvg(d, style);
  const D = (points, style) => k.dotsSvg(points, style);
  const at = (point) => k.iso(point, P);
  const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
  return { S, L, D, at, byDepth };
}

const axisX = (o) => G.frameOf(o, [1, 0, 0], [0, 1, 0], [0, 0, 1]);
const axisY = (o) => G.frameOf(o, [0, 1, 0], [1, 0, 0], [0, 0, 1]);

/** A straight round bar between two world points, as a solid. */
function bar(a, b, r, P, options = {}) {
  const length = dist3(a, b);
  const dir = [(b[0] - a[0]) / length, (b[1] - a[1]) / length, (b[2] - a[2]) / length];
  return G.disc(0, length, r, G.frameAlong(a, dir), P, options);
}

/* ---------- items ---------- */

/**
 * The ordered list of things drawn, back to front. Each item: id, xform (which pose shift moves
 * it), group (what the page can highlight), cls (extra class on its group), svg (static markup)
 * or live(pose) with deps (the pose fields it depends on).
 */
export function items(P) {
  const { S, L, D, at, byDepth } = toolkit(P);
  const list = [];
  const put = (item) => list.push(item);
  const FW = axisX([0, 0, ZC0]);

  /* --- rig base --- */
  put({
    id: "base",
    group: "rig",
    svg: (() => {
      const out = [`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT_H, BASE_H, P)}"/>`];
      for (const [x, y] of byDepth(k.corners(BASE, 26))) {
        out.push(S(k.cylinder(x, y, 9, -BASE_H - FOOT_H, 2, P, 28), { tone: "lo" }));
        out.push(S(k.cylinder(x, y, 7, -BASE_H - FOOT_H + 2, FOOT_H - 2, P, 28), { tone: "mid" }));
      }
      out.push(S(k.slabOf(BASE, -BASE_H, BASE_H, P, 8, 1.6), { tone: "mid" }));
      out.push(L(k.planOutline(k.insetPlan(BASE, 6), 0, P), { tone: "faint" }));
      const screws = k.corners(BASE, 12);
      out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.6, 0, P, 16)).join(""), { tone: "lo" }));
      out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
      const strip = { x: 20, y: BASE.y + BASE.d - 22, w: 280, d: 9, r: 1 };
      out.push(S(k.slabOf(strip, 0, 0.7, P, 3), { tone: "lo", crease: "none" }));
      const ticks = k.topTicks(strip.x + 5, strip.x + strip.w - 5, 5, 5, strip.y + 1, 0.7, [2.6, 4.4], P);
      out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
      out.push(D([at([strip.x + 2.5, strip.y + 4.5, 0.7]), at([strip.x + strip.w - 2.5, strip.y + 4.5, 0.7])], { size: 0.45 }));
      const label = { x: 190, y: BASE.y + 22, w: 88, d: 26, r: 2 };
      out.push(S(k.slabOf(label, 0, 0.9, P, 4), { tone: "lo", crease: "none" }));
      out.push(L(k.planOutline(k.insetPlan(label, 2.4), 0.9, P), { tone: "faint" }));
      [[8, 20, 70], [8, 13, 46], [8, 6, 58]].forEach(([dx, dy, len]) => out.push(L(k.lineOnTop([label.x + dx, label.y + dy], [label.x + dx + len, label.y + dy], 0.9, P), { tone: "lo", free: true })));
      out.push(D([at([label.x + 5, label.y + 5, 0.9]), at([label.x + label.w - 5, label.y + label.d - 5, 0.9])], { size: 0.45 }));
      return out.join("");
    })(),
  });

  /* --- the grip circle, engraved on a plate at the front left of the base --- */
  const GRIP = [-84, 150];
  const GRIP_R = 40;
  put({
    id: "gripplate",
    group: "grip",
    svg: (() => {
      const plate = { x: GRIP[0] - 54, y: GRIP[1] - 54, w: 108, d: 108, r: 4 };
      const out = [];
      out.push(S(k.slabOf(plate, 0, 1.4, P, 4, 0.3), { tone: "lo", crease: "none" }));
      out.push(L(k.ring(GRIP[0], GRIP[1], GRIP_R, 1.4, P, 56), { tone: "mid" }));
      out.push(L(k.ring(GRIP[0], GRIP[1], GRIP_R * 0.5, 1.4, P, 40), { tone: "faint" }));
      out.push(L(k.lineOnTop([GRIP[0] - GRIP_R, GRIP[1]], [GRIP[0] + GRIP_R, GRIP[1]], 1.4, P), { tone: "faint" }));
      out.push(L(k.lineOnTop([GRIP[0], GRIP[1] - GRIP_R], [GRIP[0], GRIP[1] + GRIP_R], 1.4, P), { tone: "faint" }));
      out.push(D(k.corners(plate, 4.5).map(([x, y]) => at([x, y, 1.4])), { size: 0.45 }));
      return out.join("");
    })(),
  });
  put({
    id: "gripdot",
    deps: ["fxs", "fys", "usage"],
    group: "grip",
    live: (pose) => {
      // braking runs up the circle, cornering across it, each as a share of what the tyre can give
      const fx = Math.max(-1.2, Math.min(1.2, pose.fxs));
      const fy = Math.max(-1.2, Math.min(1.2, pose.fys));
      const mag = Math.hypot(fx, fy);
      const shrink = mag > 1 ? 1 / mag : 1;
      const px = GRIP[0] - fy * shrink * GRIP_R;
      const py = GRIP[1] - fx * shrink * GRIP_R;
      const over = pose.usage >= 0.97;
      const tone = over ? "red" : "lit";
      return L(k.lineOnTop([GRIP[0], GRIP[1]], [px, py], 1.6, P), { tone, free: "start" }) + L(k.ring(px, py, 3, 1.6, P, 18), { tone }) + D([at([px, py, 1.6])], { size: 1, tone: "hi" });
    },
  });

  /* --- the road: a pad on two rams --- */
  put({
    id: "actuators",
    deps: ["padShift"],
    group: "rig",
    live: (pose) => {
      const out = [];
      ACT.forEach(([x, y]) => {
        out.push(S(k.cylinder(x, y, 21, 0, 26, P, 36, 0.6), { tone: "mid" }));
        out.push(L(k.sideArc(x, y, 21, 17, P, 20), { tone: "faint" }));
        out.push(L(k.ring(x, y, 12, 26, P, 24), { tone: "lo" }));
        out.push(S(k.cylinder(x, y, 7.5, 26, Math.max(0.5, PAD0 - PAD_H + pose.padShift - 26), P, 24), { tone: "hi", crease: "none" }));
      });
      return out.join("");
    },
  });
  put({
    id: "pad",
    xform: "pad",
    group: "pad",
    svg: (() => {
      const out = [`<path class="iso-halo" d="${k.haloOf(PAD, PAD0 - PAD_H, 1, P)}" style="filter:blur(5px)"/>`];
      out.push(S(k.slabOf(PAD, PAD0 - PAD_H, PAD_H, P, 8, 0.9), { tone: "mid" }));
      out.push(L(k.planOutline(k.insetPlan(PAD, 6), PAD0, P, 6), { tone: "faint" }));
      out.push(S(k.slabOf({ x: PAD.x + 10, y: PAD.y + PAD.d - 15, w: PAD.w - 20, d: 8, r: 1 }, PAD0, 0.6, P, 3), { tone: "lo", crease: "none" }));
      const t2 = k.topTicks(PAD.x + 14, PAD.x + PAD.w - 14, 6, 5, PAD.y + PAD.d - 14.2, PAD0 + 0.6, [2.2, 3.8], P);
      out.push(L(t2.minor, { tone: "lo" }), L(t2.major, { tone: "mid" }));
      out.push(D(k.corners(PAD, 7).map(([x, y]) => at([x, y, PAD0])), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({
    id: "patch",
    xform: "pad",
    deps: ["patch"],
    group: "tyre",
    live: (pose) => {
      // the contact patch: as long as the tyre is flat on the pad, as wide as the tread
      const half = Math.max(2, pose.patch / U / 2);
      const wide = 62;
      const plan = { x: -wide, y: -half, w: wide * 2, d: half * 2, r: Math.min(8, half) };
      return S(k.slabOf(plan, PAD0, 0.5, P, 6), { tone: "lit", crease: "none", flat: true, lit: true });
    },
  });

  /* --- inside the wheel: cavity, hub plate, disc, bell, caliper. They sit behind the tyre,
         which is drawn after them with a window cut at the open end of the rim. --- */
  put({
    id: "cavity",
    xform: "wheel",
    group: "brake",
    svg: `<path d="${G.circleOf(FW, OPEN_X, R_OPEN, P)}" style="fill:var(--anatomy-shade-0);stroke:none"/>`,
  });
  put({
    id: "hub-plate",
    xform: "wheel",
    group: "brake",
    svg: S(G.solidOf([[0, 98], [6, 98], [6, 92], [20, 92], [20, 40], [28, 40], [28, 14], [60, 14]], FW, P, { smooth: false }), { tone: "lo", crease: "faint" }),
  });
  const DISC_S = [28, 44];
  put({
    id: "disc",
    xform: "wheel",
    group: "brake",
    svg: S(G.ringBand(DISC_S[0], DISC_S[1], 40, 69.5, FW, P, { bevel: 0.3 }), { tone: "hi", crease: "lo" }) + L(G.circleOf(FW, DISC_S[1], 62, P), { tone: "faint" }),
  });
  put({
    id: "disc-heat",
    xform: "wheel",
    deps: ["glowAmount"],
    group: "brake",
    live: (pose) => {
      if (pose.glowAmount < 0.02) return "";
      const g = pose.glowAmount;
      const r = 235 + Math.round(20 * g);
      const gr = Math.round(70 + 150 * g * g);
      const b = Math.round(40 + 70 * g * g * g);
      const d = G.circleOf(FW, DISC_S[1] + 0.05, 69.2, P) + G.circleOf(FW, DISC_S[1] + 0.05, 41, P);
      return `<path d="${d}" fill-rule="evenodd" style="fill:rgb(${r},${gr},${b});opacity:${(0.3 + 0.6 * g).toFixed(3)};stroke:none"/>`;
    },
  });
  put({
    id: "disc-holes",
    xform: "wheel",
    deps: ["spin"],
    group: "brake",
    live: (pose) => {
      const phase = -pose.spin;
      const rows = [
        [47, 20],
        [53.5, 24],
        [60, 28],
        [66, 32],
      ];
      const out = [];
      rows.forEach(([r, count], index) => out.push(D(G.dotsOf(FW, DISC_S[1], r, count, P, { phase: phase + index * 0.07, all: true }), { size: 0.62, tone: "hi" })));
      out.push(L(G.spokesOf(FW, DISC_S[1], 41, 46, 12, P, { phase }), { tone: "lo", free: true }));
      return out.join("");
    },
  });
  put({
    id: "bell",
    xform: "wheel",
    group: "brake",
    svg: S(G.solidOf([[34, 40], [60, 40], [60, 33], [64, 33], [64, 20], [74, 20]], FW, P, { smooth: false }), { tone: "mid", crease: "faint" }),
  });
  put({
    id: "bell-bolts",
    xform: "wheel",
    deps: ["spin"],
    group: "brake",
    live: (pose) => D(G.dotsOf(FW, 64, 27, 8, P, { phase: -pose.spin, all: true }), { size: 0.7, tone: "hi" }) + L(G.circleOf(FW, 64, 12, P) + G.circleOf(FW, 74, 8, P), { tone: "lo" }),
  });
  put({
    id: "caliper",
    xform: "wheel",
    group: "brake",
    svg: (() => {
      const out = [];
      const outer = [[58, -24], [88, -19], [88, 19], [58, 24]];
      out.push(S(G.prismOf(FW, outer, 18, 28, P), { tone: "mid" }));
      out.push(S(G.prismOf(FW, [[74, -22], [90, -20], [90, 20], [74, 22]], 18, 56, P), { tone: "mid" }));
      out.push(S(G.prismOf(FW, outer, 44, 56, P), { tone: "hi", crease: "lo" }));
      [-14, 0, 14].flatMap((v) => [64, 77].map((u) => [u, v])).forEach(([u, v]) => {
        const f = axisX([56, u, ZC0 + v]);
        out.push(L(G.circleOf(f, 0, 5.4, P), { tone: "lo" }));
        out.push(L(G.circleOf(f, 0, 3.2, P), { tone: "faint" }));
      });
      out.push(S(G.disc(56, 64, 2, axisX([0, 83, ZC0 + 15]), P, { steps: 12 }), { tone: "hi", crease: "none" }));
      return out.join("");
    })(),
  });

  /* --- the tyre, drawn over the inside with a window at the open end of the rim --- */
  const TYRE_PROFILE = [
    [-74, 116], [-79, 128, 1], [-81, 142, 1], [-76, 156, 1], [-65, 163.4, 1], [-52, 165],
    [52, 165], [65, 163.4, 1], [76, 156, 1], [81, 142, 1], [79, 128, 1], [74, 116],
  ];
  put({
    id: "tyre",
    xform: "wheel",
    deps: ["ghost"],
    group: "tyre",
    cls: "fa-rubber",
    holed: true,
    svg: (() => {
      const out = [];
      out.push(S(G.lathe(TYRE_PROFILE, FW, P, { smooth: true }), { tone: "hi", crease: "faint" }));
      out.push(L(G.circleOf(FW, 81, 136, P) + G.circleOf(FW, 79.6, 150, P), { tone: "faint" }));
      out.push(L(G.arcOf(FW, 52.5, 165, P, { slope: 0 }) + G.arcOf(FW, -52.5, 165, P, { slope: 0 }), { tone: "faint" }));
      return out.join("");
    })(),
  });
  put({
    id: "tyre-marks",
    xform: "wheel",
    deps: ["spin", "ghost"],
    group: "tyre",
    cls: "fa-rubber",
    live: (pose) => {
      const phase = -pose.spin;
      const ticks = G.spokesOf(FW, 80.6, 132, 138, 48, P, { phase });
      const stripe = G.arcOf(FW, 80.8, 146, P, { from: phase + 0.4, to: phase + 1.5, steps: 24 });
      return L(ticks, { tone: "lo", free: true }) + `<path class="iso-line" data-free="" style="stroke:var(--fa-compound);stroke-width:1.8" d="${stripe}"/>`;
    },
  });
  put({
    id: "rim-lip",
    xform: "wheel",
    group: "wheel",
    holed: true,
    svg: (() => {
      const out = [];
      out.push(S(G.ringBand(72, OPEN_X, R_OPEN, 124, FW, P, { bevel: 0.5 }), { tone: "mid" }));
      out.push(L(G.circleOf(FW, OPEN_X, 112, P), { tone: "faint" }));
      out.push(D(G.dotsOf(FW, OPEN_X, 118, 24, P, { all: true }), { size: 0.5, tone: "mid" }));
      return out.join("");
    })(),
  });

  /* --- upright, hub nut and joints are outside the wheel --- */
  put({
    id: "upright",
    xform: "wheel",
    group: "suspension",
    svg: (() => {
      const out = [];
      const body = { x: 88, y: -17, w: 26, d: 34, r: 6 };
      out.push(S(k.slabOf(body, ZC0 - 60, 120, P, 6, 0.9), { tone: "mid" }));
      out.push(L(k.planOutline(k.insetPlan(body, 5), ZC0 + 60, P, 5), { tone: "faint" }));
      const nut = axisX([114, 0, ZC0]);
      out.push(S(G.disc(0, 8, 14, nut, P), { tone: "hi", crease: "lo" }));
      out.push(L(G.ribsOf(nut, 0, 8, 14, 20, P, { seams: true }), { tone: "lo" }));
      out.push(D(G.dotsOf(nut, 8, 8, 6, P, { all: true }), { size: 0.5 }));
      out.push(S(k.slabOf({ x: 98, y: 12, w: 14, d: 28, r: 2 }, ZC0 - 56, 8, P, 3, 0.4), { tone: "mid" }));
      out.push(S(k.slabOf({ x: 100, y: -46, w: 14, d: 30, r: 2 }, ZC0 + 3, 7, P, 3, 0.4), { tone: "mid" }));
      return out.join("");
    })(),
  });

  /* --- suspension links, rebuilt from their true end points --- */
  const ball = (c, r) => S(G.sphereOf(c, r, P, { steps: 28 }), { tone: "hi", crease: "none" });
  put({
    id: "arms-lower",
    deps: ["zc", "chassisShift"],
    group: "suspension",
    live: (pose) => {
      const out = [];
      const outer = add([0, 0, pose.zc], BALL_LOW);
      const ins = LOWER_IN.map((p) => add(p, [0, 0, pose.chassisShift]));
      byDepth(ins).forEach((inner) => out.push(S(bar(inner, outer, 3.6, P, { steps: 20 }), { tone: "hi", crease: "none" })));
      out.push(ball(outer, 6.4));
      ins.forEach((p) => out.push(ball(p, 4.6)));
      return out.join("");
    },
  });
  put({
    id: "arms-upper",
    deps: ["zc", "chassisShift"],
    group: "suspension",
    live: (pose) => {
      const out = [];
      const outer = add([0, 0, pose.zc], BALL_UP);
      const ins = UPPER_IN.map((p) => add(p, [0, 0, pose.chassisShift]));
      byDepth(ins).forEach((inner) => out.push(S(bar(inner, outer, 3.2, P, { steps: 20 }), { tone: "hi", crease: "none" })));
      out.push(ball(outer, 5.8));
      ins.forEach((p) => out.push(ball(p, 4.2)));
      return out.join("");
    },
  });
  put({
    id: "track-rod",
    deps: ["zc", "chassisShift"],
    group: "suspension",
    live: (pose) => {
      const outer = add([0, 0, pose.zc], TRACK_OUT);
      const inner = add(TRACK_IN, [0, 0, pose.chassisShift]);
      return S(bar(inner, outer, 2.4, P, { steps: 16 }), { tone: "mid", crease: "none" }) + ball(outer, 4) + ball(inner, 3.6);
    },
  });
  put({
    id: "pushrod",
    deps: ["zc", "chassisShift", "rocker"],
    group: "pushrod",
    live: (pose) => {
      const a = add([0, 0, pose.zc], PUSH_OUT);
      const r = pose.rocker;
      const b = [PIVOT[0] + ROCK_A * Math.cos(r), PIVOT[1], PIVOT[2] + pose.chassisShift + ROCK_A * Math.sin(r)];
      return S(bar(a, b, 4, P, { steps: 20 }), { tone: "lit", crease: "none", lit: true }) + ball(a, 5.4) + ball(b, 5.4);
    },
  });

  /* --- the chassis stub: wall, torsion bar, rocker, damper (they move with the sprung mass) --- */
  put({
    id: "chassis-wall",
    xform: "chassis",
    group: "chassis",
    svg: (() => {
      const out = [];
      out.push(S(k.slabOf(WALL, WALL_BOTTOM, WALL_TOP - WALL_BOTTOM, P, 5, 1), { tone: "mid" }));
      out.push(L(k.sideSeam(WALL, WALL_BOTTOM + 22, P, 5), { tone: "faint" }));
      out.push(L(k.sideSeam(WALL, WALL_TOP - 22, P, 5), { tone: "faint" }));
      // a panel joint on the front face, bottom edge to top edge
      const jx = WALL.x + WALL.w * 0.55;
      out.push(L(k.segment([jx, WALL.y + WALL.d, WALL_BOTTOM], [jx, WALL.y + WALL.d, WALL_TOP], P), { tone: "faint" }));
      out.push(D([0.2, 0.4, 0.62, 0.84].map((f) => at([WALL.x + WALL.w * f, WALL.y + WALL.d, WALL_TOP - 11])), { size: 0.5 }));
      out.push(L(k.planOutline(k.insetPlan(WALL, 8), WALL_TOP, P, 5), { tone: "faint" }));
      out.push(D(k.corners(WALL, 8).map(([x, y]) => at([x, y, WALL_TOP])), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({
    id: "torsion",
    xform: "chassis",
    deps: ["rocker"],
    group: "torsion",
    live: (pose) => {
      const twist = ROCK_A0 - pose.rocker;
      const out = [];
      const F = axisY([PIVOT[0], 0, PIVOT[2]]);
      [-26, 70].forEach((y) => out.push(S(k.slabOf({ x: PIVOT[0] - 12, y: y - 7, w: 24, d: 14, r: 2 }, WALL_TOP, PIVOT[2] - WALL_TOP + 8, P, 3, 0.5), { tone: "mid" })));
      out.push(S(G.disc(-26, 70, 7.5, F, P, { steps: 28 }), { tone: "hi", crease: "none" }));
      // splines wind up as the rocker turns; both ends are held in a pedestal
      const turns = twist * 2.4;
      out.push(L(G.ribsOf(F, -20, 24, 7.5, 12, P, { twist: turns, seams: true }), { tone: "lo" }));
      out.push(L(G.ribsOf(F, 44, 64, 7.5, 12, P, { twist: -turns * 0.9, seams: true }), { tone: "lo" }));
      return out.join("");
    },
  });
  put({
    id: "rocker",
    xform: "chassis",
    deps: ["rocker"],
    group: "pushrod",
    live: (pose) => {
      const out = [];
      const r = pose.rocker;
      const F = G.frameOf([PIVOT[0], Y0 - 4, PIVOT[2]], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
      const bDir = r + (ROCK_B0 - ROCK_A0);
      const wing = (ang, len) => {
        const n = [-Math.sin(ang), Math.cos(ang)];
        const tip = [len * Math.cos(ang), len * Math.sin(ang)];
        return [[n[0] * 11, n[1] * 11], [tip[0] + n[0] * 6, tip[1] + n[1] * 6], [tip[0] - n[0] * 6, tip[1] - n[1] * 6], [-n[0] * 11, -n[1] * 11]];
      };
      out.push(S(G.prismOf(F, wing(r, ROCK_A), 0, 8, P), { tone: "hi", crease: "lo" }));
      out.push(S(G.prismOf(F, wing(bDir, ROCK_B), 0, 8, P), { tone: "hi", crease: "lo" }));
      out.push(S(G.disc(-3, 11, 12.5, F, P, { steps: 30 }), { tone: "mid" }));
      out.push(L(G.circleOf(F, 11, 7.5, P), { tone: "lo" }));
      out.push(D([at([PIVOT[0], Y0 + 7, PIVOT[2]])], { size: 0.7, tone: "hi" }));
      return out.join("");
    },
  });
  put({
    id: "damper",
    xform: "chassis",
    deps: ["rocker"],
    group: "damper",
    live: (pose) => {
      const out = [];
      const bDir = pose.rocker + (ROCK_B0 - ROCK_A0);
      const tip = [PIVOT[0] + ROCK_B * Math.cos(bDir), Y0, PIVOT[2] + ROCK_B * Math.sin(bDir)];
      const m = DAMPER_MOUNT;
      const len = dist3(m, tip);
      const dir = [(tip[0] - m[0]) / len, (tip[1] - m[1]) / len, (tip[2] - m[2]) / len];
      const F = G.frameAlong(m, dir);
      out.push(S(k.slabOf({ x: m[0] - 11, y: m[1] - 11, w: 22, d: 22, r: 2 }, WALL_TOP, m[2] - WALL_TOP - 6, P, 3, 0.4), { tone: "mid" }));
      const bodyLen = 36;
      out.push(S(G.disc(0, bodyLen, 9, F, P, { steps: 28 }), { tone: "hi", crease: "lo" }));
      out.push(L(G.arcOf(F, 9, 9, P) + G.arcOf(F, 27, 9, P), { tone: "faint" }));
      if (len > bodyLen) out.push(S(G.disc(bodyLen, len - 4, 3.8, F, P, { steps: 18 }), { tone: "lit", crease: "none", lit: true }));
      out.push(ball(tip, 5));
      return out.join("");
    },
  });
  put({
    id: "guide-lower",
    deps: ["chassisShift"],
    group: "rig",
    live: (pose) => ROD_Y.map((y) => S(k.cylinder(ROD_X, y, 4.5, 0, Math.max(1, WALL_BOTTOM + pose.chassisShift + 2), P, 20), { tone: "mid", crease: "none" })).join(""),
  });
  put({
    id: "guide-blocks",
    xform: "chassis",
    group: "chassis",
    svg: ROD_Y.map((y) => S(k.slabOf({ x: ROD_X - 14, y: y - 13, w: 28, d: 26, r: 2 }, WALL_BOTTOM + 4, BLOCK_H, P, 4, 0.6), { tone: "mid" })).join(""),
  });
  put({
    id: "guide-upper",
    deps: ["chassisShift"],
    group: "rig",
    live: (pose) =>
      ROD_Y.map((y) => {
        const z0 = WALL_BOTTOM + 4 + BLOCK_H + pose.chassisShift;
        return S(k.cylinder(ROD_X, y, 4.5, z0, WALL_TOP + 70 - z0, P, 20), { tone: "mid", crease: "none" }) + S(k.cylinder(ROD_X, y, 7, WALL_TOP + 70, 5, P, 20), { tone: "hi" });
      }).join(""),
  });
  return list;
}

/** Markup of one item at a pose (static items ignore the pose). */
export const markup = (item, pose) => (item.live ? item.live(pose) : item.svg || "");

/** How far an item sits from where it was built, as a world offset [dx, dy, dz]. */
export function shiftOf(item, pose) {
  if (item.xform === "wheel") return [0, 0, pose.wheelShift];
  if (item.xform === "chassis") return [0, 0, pose.chassisShift];
  if (item.xform === "pad") return [0, 0, pose.padShift];
  return [0, 0, 0];
}

/** The window cut in the tyre at the open end of the rim, so the brake shows through it. */
export function windowDefs(P) {
  const FW = axisX([0, 0, ZC0]);
  const hole = G.circleOf(FW, OPEN_X, R_OPEN, P);
  return `<defs><clipPath id="tyre-window"><path clip-rule="evenodd" d="M-4000 -4000H4000V4000H-4000Z${hole}"/></clipPath></defs>`;
}

// Geometry of the gearbox bench: an eight-speed sequential gearbox in cutaway beside a detachable
// steering wheel. World units are 2 mm; x runs along the gearbox, y across it, z up.
// Runs in Node (the build draws the page) and in the browser (the live page redraws what moves).
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as G from "../../vendor/kit/lathe.mjs";
import * as M from "./drivetrain.model.mjs";

export const WIDTH = 960;
export const HEIGHT = 580;

export const BASE = { x: -250, y: -135, w: 662, d: 292, r: 12 };
const BASE_H = 12;
const FOOT_H = 6;

const Z_AX = 70; // shaft axis height
const Y_MAIN = -28;
const Y_LAY = 28;
const X0 = 22;
const PITCH = 34;
const GEAR_T = 12;
const CENTRE = 56; // distance between the shafts
const CASE = { x: -6, y: -120, w: 312, d: 238, r: 4 };
const FLOOR_TOP = 20;
const CASE_TOP = 150;
const DECK_Z = 134; // fork rail height; the arms run just under it
const DRUM = { y: -92, z: 104, r: 13, x0: 8, x1: 292 };

export const gearX = (i) => X0 + PITCH * (i - 1);
export const mainRadius = (i) => CENTRE / (1 + M.pairRatio(i));
export const layRadius = (i) => CENTRE - mainRadius(i);
export const ringX = (kk) => gearX(2 * kk + 1) + PITCH / 2;
const RING_TRAVEL = 6;

// the steering wheel and its stand, to the front left
const WHEEL = { x: -150, y: 40, z: 112 }; // centre of the wheel plate, face at y
const WHEEL_SLIDE = 96; // how far it comes off the column
const WS = 1.35; // the wheel is drawn this much bigger than its coordinates say
const wp = (u, v, dy = 0) => [WHEEL.x + u * WS, WHEEL.y + dy, WHEEL.z + v * WS];
const KNOBS = [
  { id: "bb", u: -38, v: 10 },
  { id: "diff", u: 38, v: 10 },
  { id: "mode", u: 0, v: -16 },
];

export function projection() {
  const pts = [
    ...k.boxCorners(BASE, -BASE_H - FOOT_H, 0),
    [WHEEL.x - 120, WHEEL.y + WHEEL_SLIDE, WHEEL.z + 60],
    [330, 0, CASE_TOP + 30],
  ];
  return k.fitProjection(pts, WIDTH, HEIGHT, { pad: 24, azimuth: 56, elevation: 33 });
}

export const REST_POSE = () => poseOf({ gear: 1, rpm: 4000, speed: 0, phase: 0 });

/** Everything that moves, as numbers the items depend on. `phase` is the displayed crank angle. */
export function poseOf({ gear, rpm, speed, phase, shaftPhase = 0, drum = 1, rings = null, detach = 0, padU = 0, padD = 0, bb = 3, diff = 1, mode = 2, limiter = 0, flash = 0, leds = 0 }) {
  const pose = { gear, rpm, speed, pMain: phase, pShaft: shaftPhase, drum, detach, padU, padD, bb, diff, mode, limiter, flash, leds };
  for (let i = 1; i <= 8; i++) pose["p" + i] = phase / M.pairRatio(i);
  const r = rings || M.ringsFor(gear);
  for (let kk = 0; kk < 4; kk++) pose["r" + kk] = r[kk];
  return pose;
}

export function toolkit(P) {
  const S = (paths, style) => k.solidSvg(paths, style);
  const L = (d, style) => k.lineSvg(d, style);
  const D = (points, style) => k.dotsSvg(points, style);
  const at = (point) => k.iso(point, P);
  return { S, L, D, at };
}

const frameX = (x, y, z) => G.frameOf([x, y, z], [1, 0, 0], [0, 1, 0], [0, 0, 1]);
const frameY = (x, y, z) => G.frameOf([x, y, z], [0, 1, 0], [1, 0, 0], [0, 0, 1]);

/* ---------- items ---------- */

export function items(P) {
  const { S, L, D, at } = toolkit(P);
  const list = [];
  const put = (item) => list.push(item);
  const ball = (c, r) => S(G.sphereOf(c, r, P, { steps: 24 }), { tone: "hi", crease: "none" });

  /* --- base --- */
  put({
    id: "base",
    group: "rig",
    svg: (() => {
      const out = [`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT_H, BASE_H, P)}"/>`];
      const byDepth = (pts) => [...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
      for (const [x, y] of byDepth(k.corners(BASE, 26))) {
        out.push(S(k.cylinder(x, y, 9, -BASE_H - FOOT_H, 2, P, 28), { tone: "lo" }));
        out.push(S(k.cylinder(x, y, 7, -BASE_H - FOOT_H + 2, FOOT_H - 2, P, 28), { tone: "mid" }));
      }
      out.push(S(k.slabOf(BASE, -BASE_H, BASE_H, P, 8, 1.6), { tone: "mid" }));
      out.push(L(k.planOutline(k.insetPlan(BASE, 6), 0, P), { tone: "faint" }));
      const screws = k.corners(BASE, 12);
      out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.6, 0, P, 16)).join(""), { tone: "lo" }));
      out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
      const strip = { x: 30, y: BASE.y + BASE.d - 22, w: 300, d: 9, r: 1 };
      out.push(S(k.slabOf(strip, 0, 0.7, P, 3), { tone: "lo", crease: "none" }));
      const ticks = k.topTicks(strip.x + 5, strip.x + strip.w - 5, 5, 5, strip.y + 1, 0.7, [2.6, 4.4], P);
      out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
      out.push(D([at([strip.x + 2.5, strip.y + 4.5, 0.7]), at([strip.x + strip.w - 2.5, strip.y + 4.5, 0.7])], { size: 0.45 }));
      return out.join("");
    })(),
  });

  /* ===================== steering wheel and its stand ===================== */
  put({
    id: "stand",
    group: "wheel",
    svg: (() => {
      const out = [];
      const foot = { x: WHEEL.x - 36, y: -34, w: 72, d: 72, r: 5 };
      out.push(`<path class="iso-halo" d="${k.haloOf(foot, 0, 6, P)}" style="filter:blur(5px)"/>`);
      out.push(S(k.slabOf(foot, 0, 8, P, 5, 0.8), { tone: "mid" }));
      out.push(D(k.corners(foot, 8).map(([x, y]) => at([x, y, 8])), { size: 0.5 }));
      out.push(S(k.cylinder(WHEEL.x, -6, 12, 8, 6, P, 28), { tone: "mid" }));
      out.push(S(k.cylinder(WHEEL.x, -6, 9, 14, WHEEL.z - 14 - 6, P, 24), { tone: "hi", crease: "none" }));
      out.push(L([40, 70].map((z) => k.sideArc(WHEEL.x, -6, 9, z, P, 14)).join(""), { tone: "faint" }));
      const F = G.frameOf([WHEEL.x, -6, WHEEL.z], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
      out.push(S(G.disc(-10, 0, 12, F, P), { tone: "mid", crease: "none" }));
      out.push(S(G.disc(0, WHEEL.y - 34, 9, F, P), { tone: "hi", crease: "none" }));
      return out.join("");
    })(),
  });
  put({
    id: "quick-release",
    group: "wheel",
    svg: (() => {
      // the fixed half of the quick release: a collar with a spring-loaded ring and three pins
      const F = G.frameOf([WHEEL.x, WHEEL.y - 34, WHEEL.z], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
      const out = [];
      out.push(S(G.disc(0, 12, 17, F, P), { tone: "mid", crease: "lo" }));
      out.push(L(G.ribsOf(F, 1, 11, 17, 24, P, { seams: true }), { tone: "lo" }));
      out.push(S(G.disc(12, 22, 12.5, F, P), { tone: "hi", crease: "lo" }));
      out.push(D(G.dotsOf(F, 22, 8, 3, P, { all: true, phase: 0.4 }), { size: 0.9, tone: "hi" }));
      return out.join("");
    })(),
  });
  put({
    id: "wheel-spigot",
    group: "wheel",
    xform: "wheel",
    svg: (() => {
      const F = G.frameOf([WHEEL.x, WHEEL.y - 10, WHEEL.z], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
      return S(G.disc(-24, 0, 12, F, P, { steps: 28 }), { tone: "mid", crease: "lo" }) + L(G.circleOf(F, -24, 6.5, P), { tone: "faint" });
    })(),
  });
  // the two paddles hang behind the plate, their blades reaching out past its sides
  const paddle = (side) => (pose) => {
    const F = G.frameOf([WHEEL.x, WHEEL.y - 18, WHEEL.z], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
    const press = side < 0 ? pose.padD : pose.padU;
    const dv = press * 4;
    const blade = [[56, -14], [66, -10], [92, -58 - dv], [83, -63 - dv]].map(([u, v]) => [side * u * WS, v * WS]);
    const on = press > 0.3;
    return S(G.prismOf(F, blade, 0, 4, P), { tone: on ? "lit" : "hi", lit: on, crease: "lo" }) + D([at(wp(side * 86, -56 - dv, -16))], { size: 0.6, tone: "hi" });
  };
  put({ id: "paddle-down", group: "wheel", xform: "wheel", deps: ["padD"], live: paddle(-1) });
  put({ id: "paddle-up", group: "wheel", xform: "wheel", deps: ["padU"], live: paddle(1) });
  put({
    id: "wheel-body",
    group: "wheel",
    xform: "wheel",
    svg: (() => {
      const out = [];
      const F = G.frameOf([WHEEL.x, WHEEL.y - 10, WHEEL.z], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
      const sc = (poly) => poly.map(([u, v]) => [u * WS, v * WS]);
      out.push(S(G.prismOf(F, sc([[-50, -26], [50, -26], [50, 26], [-50, 26]]), 0, 10, P), { tone: "hi", crease: "lo" }));
      out.push(S(G.prismOf(F, sc([[-48, -16], [-62, -12], [-72, -40], [-62, -54], [-46, -52], [-42, -24]]), 0, 10, P), { tone: "mid", crease: "lo" }));
      out.push(S(G.prismOf(F, sc([[48, -16], [62, -12], [72, -40], [62, -54], [46, -52], [42, -24]]), 0, 10, P), { tone: "mid", crease: "lo" }));
      // grips: ribs across the two wings, from the plate edge to the outer edge of the wing
      [-1, 1].forEach((s) => {
        for (let n = 0; n < 5; n++) {
          const v = -20 - n * 6.2;
          out.push(L(k.segment(wp(s * (46 + n * 1.4), v, 0), wp(s * (66 + n * 1.4), v - 5, 0), P), { tone: "faint" }));
        }
      });
      out.push(D([[-45, 22], [45, 22], [-45, -22], [45, -22]].map(([u, v]) => at(wp(u, v))), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({
    id: "wheel-display",
    group: "wheel",
    xform: "wheel",
    deps: ["gear", "speed", "rpm", "leds", "limiter", "detach"],
    live: (pose) => {
      const out = [];
      const M2 = k.sideMatrix([WHEEL.x, WHEEL.y], WHEEL.z, P, "left");
      // the screen: a recess drawn in the plane of the wheel face (u across, v down)
      const u0 = -27 * WS, u1 = 27 * WS, v0 = -23 * WS, v1 = 3 * WS;
      out.push(`<g transform="${M2}"><rect x="${u0}" y="${v0}" width="${u1 - u0}" height="${v1 - v0}" rx="3" style="fill:var(--anatomy-shade-0);stroke:var(--anatomy-mid);stroke-width:.5;vector-effect:non-scaling-stroke"/>`);
      const n = pose.detach > 0.5 ? "-" : pose.gear;
      out.push(`<text x="0" y="${v0 + 20 * WS}" text-anchor="middle" style="font:700 ${(21 * WS) / 1.2}px var(--anatomy-mono);fill:var(--anatomy-hi)">${n}</text>`);
      out.push(`<text x="${u0 + 4}" y="${v1 - 4}" style="font:600 ${4.6 * WS}px var(--anatomy-mono);fill:var(--anatomy-mid)">${Math.round(pose.speed)}</text><text x="${u1 - 4}" y="${v1 - 4}" text-anchor="end" style="font:600 ${4.6 * WS}px var(--anatomy-mono);fill:var(--anatomy-mid)">${(pose.rpm / 1000).toFixed(1)}</text></g>`);
      // rev lights across the top of the plate, filling as the engine speeds up
      const dots = [];
      for (let i = 0; i < 15; i++) {
        const lit = pose.detach < 0.5 && i < pose.leds;
        const p = at(wp(-28 + i * 4, 21.5));
        dots.push(`<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="${lit ? 1.3 : 0.8}" style="fill:${lit ? (i > 11 ? "var(--anatomy-red)" : i > 7 ? "var(--anatomy-blue)" : "var(--anatomy-green)") : "var(--anatomy-lo)"}"/>`);
      }
      out.push(dots.join(""));
      return out.join("");
    },
  });
  // three rotary switches; they turn when you set them or click them
  ["bb", "diff", "mode"].forEach((id) => {
    const kn = KNOBS.find((n) => n.id === id);
    put({
      id: "knob-" + id,
      group: "wheel",
      xform: "wheel",
      deps: [id],
      live: (pose) => {
        const F = G.frameOf(wp(kn.u, kn.v), [0, 1, 0], [1, 0, 0], [0, 0, 1]);
        const out = [];
        const steps = id === "bb" ? 7 : 3;
        const sweep = id === "bb" ? Math.PI * 1.5 : Math.PI;
        const angleOf = (i) => Math.PI / 2 + sweep / 2 - (steps > 1 ? i / (steps - 1) : 0) * sweep;
        const r = 8 * WS;
        out.push(S(G.disc(0, 5, r, F, P, { steps: 30 }), { tone: "hi", crease: "lo" }));
        out.push(L(G.ribsOf(F, 0.5, 4.5, r, 22, P, { seams: false }), { tone: "lo" }));
        const at2 = (a, rad) => at([WHEEL.x + kn.u * WS + Math.cos(a) * rad, WHEEL.y + 0.2, WHEEL.z + kn.v * WS + Math.sin(a) * rad]);
        const a0 = at2(angleOf(pose[id]), r * 0.2);
        const b0 = at2(angleOf(pose[id]), r * 0.82);
        const ticks = [];
        for (let i = 0; i < steps; i++) {
          const p0 = at2(angleOf(i), r * 1.28);
          const p1 = at2(angleOf(i), r * 1.6);
          ticks.push(`M${p0[0].toFixed(1)} ${p0[1].toFixed(1)}L${p1[0].toFixed(1)} ${p1[1].toFixed(1)}`);
        }
        out.push(L(ticks.join(""), { tone: "mid", free: true }));
        out.push(`<path class="iso-line" data-tone="lit" data-free="start" style="stroke-width:1.2" d="M${a0[0].toFixed(1)} ${a0[1].toFixed(1)}L${b0[0].toFixed(1)} ${b0[1].toFixed(1)}"/>`);
        return out.join("");
      },
    });
  });
  put({
    id: "wheel-buttons",
    group: "wheel",
    xform: "wheel",
    svg: (() => {
      const out = [];
      [[-38, -12], [38, -12], [-19, -17], [19, -17]].forEach(([u, v]) => {
        const F = G.frameOf(wp(u, v), [0, 1, 0], [1, 0, 0], [0, 0, 1]);
        out.push(S(G.disc(0, 2.6, 3.8 * WS, F, P, { steps: 18 }), { tone: "mid", crease: "none" }));
      });
      return out.join("");
    })(),
  });

  /* ===================== gearbox ===================== */
  /* --- the far end: clutch and the bell housing plate --- */
  put({
    id: "case-floor",
    group: "case",
    svg: (() => {
      const out = [`<path class="iso-halo" d="${k.haloOf(CASE, 4, FLOOR_TOP - 4, P)}"/>`];
      // the tray stands on four feet
      k.corners(CASE, 16).forEach(([x, y]) => out.push(S(k.cylinder(x, y, 8, 0, 8, P, 20), { tone: "mid", crease: "none" })));
      out.push(S(k.slabOf(CASE, 8, FLOOR_TOP - 8, P, 5, 0.9), { tone: "mid" }));
      out.push(L(k.planOutline(k.insetPlan(CASE, 8), FLOOR_TOP, P, 5), { tone: "faint" }));
      out.push(D(k.corners(CASE, 10).map(([x, y]) => at([x, y, FLOOR_TOP])), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({
    id: "clutch",
    group: "case",
    svg: (() => {
      const F = frameX(0, Y_MAIN, Z_AX);
      const out = [];
      out.push(S(G.solidOf([[-62, 3], [-62, 30], [-58, 30], [-58, 34], [-30, 34], [-30, 26], [-12, 26], [-12, 12], [-4, 12], [-4, 3]], F, P, { smooth: false }), { tone: "mid", crease: "faint" }));
      out.push(L(G.ribsOf(F, -56, -32, 34, 40, P, { seams: false, fade: null }), { tone: "lo" }));
      out.push(D(G.dotsOf(F, -62, 22, 8, P, { all: true }), { size: 0.6, tone: "hi" }));
      return out.join("");
    })(),
  });
  put({
    id: "front-plate",
    group: "case",
    svg: (() => {
      const out = [];
      const plate = { x: -6, y: CASE.y, w: 12, d: CASE.d, r: 3 };
      out.push(S(k.slabOf(plate, FLOOR_TOP, CASE_TOP - FLOOR_TOP, P, 4, 0.8), { tone: "mid" }));
      out.push(L(k.sideSeam(plate, CASE_TOP - 18, P, 4), { tone: "faint" }));
      // bearing bosses for the two shafts, bolt circles on them
      [Y_MAIN, Y_LAY].forEach((y) => {
        const F = frameX(6, y, Z_AX);
        out.push(S(G.disc(0, 6, 16, F, P), { tone: "hi", crease: "lo" }));
        out.push(D(G.dotsOf(F, 6, 12, 8, P, { all: true }), { size: 0.55, tone: "mid" }));
      });
      return out.join("");
    })(),
  });
  put({
    id: "back-wall",
    group: "case",
    svg: (() => {
      const wall = { x: -6, y: CASE.y, w: CASE.w, d: 6, r: 2 };
      const out = [S(k.slabOf(wall, FLOOR_TOP, CASE_TOP - FLOOR_TOP, P, 3, 0.7), { tone: "mid" })];
      out.push(L(k.sideSeam(wall, CASE_TOP - 18, P, 3), { tone: "faint" }));
      return out.join("");
    })(),
  });

  /* --- main shaft, layshaft and the eight gear pairs, nearest the clutch first --- */
  const mainF = frameX(0, Y_MAIN, Z_AX);
  const layF = frameX(0, Y_LAY, Z_AX);
  const bounds = (i) => [gearX(i) - GEAR_T / 2, gearX(i) + GEAR_T / 2];
  const shaftSegment = (id, F, r, s0, s1, group) =>
    put({ id, group, svg: S(G.disc(s0, s1, r, F, P, { steps: 24 }), { tone: "hi", crease: "none" }) });

  for (let i = 1; i <= 8; i++) {
    const [g0, g1] = bounds(i);
    const prevEnd = i === 1 ? -50 : bounds(i - 1)[1];
    // the shaft pieces that sit just before this pair (rings are drawn with their own pieces)
    shaftSegment(`main-shaft-${i}`, mainF, 7, prevEnd, g0, "main");
    const rm = mainRadius(i);
    put({
      id: `main-gear-${i}`,
      group: "main",
      svg: (() => {
        const F = frameX(gearX(i), Y_MAIN, Z_AX);
        const out = [];
        out.push(S(G.disc(-GEAR_T / 2, GEAR_T / 2, rm, F, P, { steps: Math.max(32, Math.round(rm * 2.6)) }), { tone: "mid", crease: "lo" }));
        out.push(L(G.circleOf(F, GEAR_T / 2, Math.max(3, rm * 0.55), P), { tone: "faint" }));
        return out.join("");
      })(),
    });
    put({
      id: `main-teeth-${i}`,
      group: "main",
      deps: ["pMain"],
      live: (pose) => {
        const F = frameX(gearX(i), Y_MAIN, Z_AX);
        const count = Math.round(rm * 1.6);
        const phase = -pose.pMain;
        return k.fadedSvg(G.ribsOf(F, -GEAR_T / 2, GEAR_T / 2, rm, count, P, { phase, fade: [0.05, 0.5], seams: true }), { tone: "lo" }) + L(G.spokesOf(F, GEAR_T / 2, 3.5, Math.max(4, rm * 0.5), 5, P, { phase }), { tone: "faint", free: true });
      },
    });

    // layshaft piece before this pair (the gap before an even gear is drawn with its dog ring), then the pair's layshaft gear
    if (i === 1 || i % 2 === 1) shaftSegment(`lay-shaft-${i}`, layF, 8, i === 1 ? 6 : bounds(i - 1)[1], g0, "lay");
    const rl = layRadius(i);
    put({
      id: `lay-gear-${i}`,
      group: "lay",
      cls: "lay-gear",
      svg: (() => {
        const F = frameX(gearX(i), Y_LAY, Z_AX);
        const out = [];
        out.push(S(G.disc(-GEAR_T / 2, GEAR_T / 2, rl, F, P, { steps: Math.max(48, Math.round(rl * 2.2)) }), { tone: "hi", crease: "lo" }));
        out.push(L(G.circleOf(F, GEAR_T / 2, Math.max(4, rl * 0.7), P), { tone: "faint" }));
        // lightening holes
        out.push(L(G.circleOf(F, GEAR_T / 2, Math.max(2.5, rl * 0.2), P), { tone: "faint" }));
        return out.join("");
      })(),
    });
    put({
      id: `lay-teeth-${i}`,
      group: "lay",
      deps: ["p" + i],
      live: (pose) => {
        const F = frameX(gearX(i), Y_LAY, Z_AX);
        const count = Math.round(rl * 1.6);
        const phase = -pose["p" + i];
        return k.fadedSvg(G.ribsOf(F, -GEAR_T / 2, GEAR_T / 2, rl, count, P, { phase, fade: [0.05, 0.5], seams: true }), { tone: "lo" }) + L(G.spokesOf(F, GEAR_T / 2, Math.max(5, rl * 0.25), Math.max(7, rl * 0.68), 6, P, { phase }), { tone: "faint", free: true });
      },
    });

    // the layshaft across the gap, then the dog ring and the fork finger and arm, after the first gear of a pair
    if (i % 2 === 1) {
      const kk = (i - 1) / 2;
      shaftSegment(`lay-gap-${kk}`, layF, 8, bounds(i)[1], bounds(i + 1)[0], "lay");
      put({
        id: `ring-${kk}`,
        group: "shift",
        deps: ["r" + kk],
        live: (pose) => {
          const x = ringX(kk) + pose["r" + kk] * RING_TRAVEL;
          const F = frameX(x, Y_LAY, Z_AX);
          const lit = Math.abs(pose["r" + kk]) > 0.5;
          const out = [];
          out.push(S(G.disc(-5, 5, 13, F, P, { steps: 36 }), { tone: lit ? "lit" : "hi", lit, crease: "lo" }));
          // dogs on both faces
          out.push(D(G.dotsOf(F, 5, 10.5, 8, P, { all: true, phase: 0.2 }), { size: 0.7, tone: "hi" }));
          out.push(L(G.circleOf(F, 0, 13.2, P), { tone: "faint" }));
          return out.join("");
        },
      });
      put({
        id: `fork-low-${kk}`,
        group: "shift",
        deps: ["r" + kk],
        live: (pose) => {
          const x = ringX(kk) + pose["r" + kk] * RING_TRAVEL;
          const out = [];
          // finger from the arm down into the ring's groove, arm back to the rail
          const armZ = DECK_Z - 12;
          out.push(S(k.slabOf({ x: x - 2.6, y: DRUM.y, w: 5.2, d: Y_LAY - DRUM.y + 2.6, r: 1 }, armZ, 4, P, 3, 0.3), { tone: "mid" }));
          out.push(S(k.slabOf({ x: x - 2.6, y: Y_LAY - 2.6, w: 5.2, d: 5.2, r: 1 }, Z_AX + 13, armZ + 4 - Z_AX - 13, P, 3, 0.3), { tone: "mid", crease: "none" }));
          out.push(S(k.slabOf({ x: x - 6, y: Y_LAY - 4, w: 12, d: 8, r: 1.5 }, Z_AX + 11, 5, P, 3, 0.3), { tone: "mid" }));
          return out.join("");
        },
      });
    }
  }
  // the end of the layshaft and the crown wheel and pinion
  shaftSegment("lay-shaft-end", layF, 8, bounds(8)[1], 326, "lay");
  put({
    id: "rear-plate",
    group: "case",
    svg: (() => {
      const plate = { x: 296, y: CASE.y, w: 10, d: CASE.d, r: 3 };
      const out = [S(k.slabOf(plate, FLOOR_TOP, 40, P, 3, 0.6), { tone: "mid" })];
      const F = frameX(306, Y_LAY, Z_AX);
      out.push(S(G.disc(0, 6, 15, F, P), { tone: "hi", crease: "lo" }));
      return out.join("");
    })(),
  });
  put({
    id: "main-shaft-end",
    group: "main",
    svg: S(G.disc(bounds(8)[1], 306, 7, mainF, P, { steps: 24 }), { tone: "hi", crease: "none" }),
  });
  put({
    id: "diff",
    group: "case",
    deps: ["pShaft"],
    svg: (() => {
      const out = [];
      // pinion on the layshaft, crown wheel across it, axle stubs out to the sides
      const pin = frameX(330, Y_LAY, Z_AX);
      out.push(S(G.solidOf([[-8, 7], [8, 12], [8, 7], [14, 7]], pin, P, { smooth: false }), { tone: "mid", crease: "lo" }));
      const crown = G.frameOf([336, Y_LAY + 4, Z_AX], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
      out.push(S(G.disc(-3, 3, 30, crown, P, { steps: 56 }), { tone: "hi", crease: "lo" }));
      out.push(S(G.disc(-26, 40, 6, crown, P, { steps: 20 }), { tone: "mid", crease: "none" }));
      out.push(L(G.circleOf(crown, 3.05, 18, P), { tone: "faint" }));
      return out.join("");
    })(),
  });
  put({
    id: "crown-teeth",
    group: "case",
    deps: ["pShaft"],
    live: (pose) => {
      const crown = G.frameOf([336, Y_LAY + 4, Z_AX], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
      return k.fadedSvg(G.ribsOf(crown, -3, 3, 30, 54, P, { phase: -pose.pShaft / M.FINAL_DRIVE, fade: [0.05, 0.5], seams: true }), { tone: "lo" }) + L(G.spokesOf(crown, 3.05, 7, 17, 6, P, { phase: -pose.pShaft / M.FINAL_DRIVE }), { tone: "faint", free: true });
    },
  });

  /* --- low front wall of the casing, then the shift drum, rail and fork tops above everything --- */
  put({
    id: "front-wall",
    group: "case",
    svg: (() => {
      const wall = { x: -6, y: CASE.y + CASE.d - 6, w: CASE.w, d: 6, r: 2 };
      const out = [S(k.slabOf(wall, FLOOR_TOP, 24, P, 3, 0.6), { tone: "mid" })];
      out.push(D([40, 120, 200, 280].map((x) => at([x, wall.y + 6, FLOOR_TOP + 14])), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({
    id: "drum-brackets",
    group: "shift",
    svg: (() => {
      const out = [];
      [DRUM.x0 - 8, DRUM.x1 + 2].forEach((x) => {
        out.push(S(k.slabOf({ x, y: DRUM.y - 14, w: 8, d: 28, r: 2 }, FLOOR_TOP, DECK_Z - FLOOR_TOP + 6, P, 3, 0.5), { tone: "mid" }));
      });
      return out.join("");
    })(),
  });
  put({
    id: "drum",
    group: "shift",
    deps: ["drum"],
    svg: (() => {
      const F = frameX(0, DRUM.y, DRUM.z);
      return S(G.disc(DRUM.x0, DRUM.x1, DRUM.r, F, P, { steps: 40 }), { tone: "hi", crease: "none" }) + S(G.disc(DRUM.x0 - 10, DRUM.x0, 5, F, P, { steps: 16 }), { tone: "mid", crease: "none" }) + S(G.disc(DRUM.x1, DRUM.x1 + 10, 5, F, P, { steps: 16 }), { tone: "mid", crease: "none" });
    })(),
  });
  put({
    id: "drum-grooves",
    group: "shift",
    deps: ["drum"],
    live: (pose) => {
      const F = frameX(0, DRUM.y, DRUM.z);
      const out = [];
      const theta = Math.PI / 2 - pose.drum * (Math.PI / 4); // the drum's frame angle that sits under the pins at the top
      for (let kk = 0; kk < 4; kk++) {
        const pts = [];
        const flags = [];
        for (let n = 0; n <= 160; n++) {
          const a = (n / 160) * (Math.PI * 2); // angle round the drum in its own frame, eight gears to a turn
          const s = ringX(kk) + M.trackAt(kk, Math.min(8, a / (Math.PI / 4))) * RING_TRAVEL;
          const world = a + theta;
          pts.push(G.pointOf(F, s, DRUM.r, world));
          flags.push(G.facingOf(F, world, P, 0) > 0.05);
        }
        G.runsOf(flags).forEach((run) => {
          if (run.length > 1) out.push(L("M" + run.map((idx) => k.iso(pts[idx], P).map((v) => v.toFixed(1)).join(" ")).join("L"), { tone: kk % 2 ? "lo" : "mid" }));
        });
      }
      return out.join("");
    },
  });
  put({
    id: "drum-detent",
    group: "shift",
    deps: ["drum"],
    live: (pose) => {
      // a star wheel on the end of the drum with eight detents, and the spring-loaded roller
      const F = frameX(DRUM.x1 + 10, DRUM.y, DRUM.z);
      const out = [];
      out.push(S(G.disc(0, 5, 17, F, P, { steps: 32 }), { tone: "mid", crease: "lo" }));
      out.push(L(G.spokesOf(F, 5, 6, 17, 8, P, { phase: Math.PI / 2 - pose.drum * (Math.PI / 4) + Math.PI / 8 }), { tone: "lo" }));
      out.push(D(G.dotsOf(F, 5, 11, 8, P, { phase: Math.PI / 2 - pose.drum * (Math.PI / 4), all: true }), { size: 0.55, tone: "hi" }));
      out.push(S(k.slabOf({ x: DRUM.x1 + 12, y: DRUM.y + 12, w: 9, d: 16, r: 2 }, DRUM.z + 8, 8, P, 3, 0.3), { tone: "mid" }));
      return out.join("");
    },
  });
  put({
    id: "rail",
    group: "shift",
    svg: S(G.disc(DRUM.x0, DRUM.x1, 3.6, G.frameOf([0, DRUM.y, DECK_Z], [1, 0, 0], [0, 1, 0], [0, 0, 1]), P, { steps: 16 }), { tone: "hi", crease: "none" }),
  });
  for (let kk = 0; kk < 4; kk++) {
    put({
      id: `fork-top-${kk}`,
      group: "shift",
      deps: ["r" + kk],
      live: (pose) => {
        const x = ringX(kk) + pose["r" + kk] * RING_TRAVEL;
        const out = [];
        const engaged = Math.abs(pose["r" + kk]) > 0.5;
        out.push(S(G.disc(0, DECK_Z - 12 - (DRUM.z + DRUM.r) + 1, 2, G.frameOf([x, DRUM.y, DRUM.z + DRUM.r - 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]), P, { steps: 12 }), { tone: "hi", crease: "none" }));
        out.push(S(k.slabOf({ x: x - 7, y: DRUM.y - 6, w: 14, d: 12, r: 2 }, DECK_Z - 12, 14, P, 3, 0.4), { tone: engaged ? "lit" : "mid", lit: engaged, crease: "lo" }));
        return out.join("");
      },
    });
  }
  return list;
}

export const markup = (item, pose) => (item.live ? item.live(pose) : item.svg || "");

/** Offset of an item from where it was built. The wheel comes off the column along y. */
export function shiftOf(item, pose) {
  if (item.xform === "wheel") return [0, WHEEL_SLIDE * pose.detach, 0];
  return [0, 0, 0];
}

/** Click targets on the stage, in viewBox units: the two paddle sides and the three rotary switches. */
export function hotspots(P, detach = 0) {
  const y = detach * WHEEL_SLIDE;
  const p = (u, v) => k.iso(wp(u, v, y), P);
  const scale = k.axisVector(P, [1, 0, 0]);
  const unit = Math.hypot(scale[0], scale[1]);
  return [
    { id: "down", at: p(-78, -40), r: 24 * unit * WS },
    { id: "up", at: p(78, -40), r: 24 * unit * WS },
    ...KNOBS.map((kn) => ({ id: kn.id, at: p(kn.u, kn.v), r: 11 * unit * WS })),
  ];
}

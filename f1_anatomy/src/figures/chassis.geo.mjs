// Geometry of the crash-test rig: the survival cell on a sled, a wall to hit, a side ram and a Halo load frame.
// World units are 5 mm. x runs forward (nose to +x), y to the near side (the cutaway side), z up.
// Runs in Node (the build) and in the browser (live redraw) through the namespaces k, G and T.
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as G from "../../vendor/kit/lathe.mjs";
import * as T from "../../vendor/kit/tube.mjs";

export const U = 0.005; // metres per unit
export const WIDTH = 960;
export const HEIGHT = 580;

export const BASE = { x: -120, y: -175, w: 780, d: 350, r: 12 };
const BASE_H = 12;
const FOOT_H = 6;
const Z0 = 18; // sled top / tub floor underside
const NOSE0 = 306; // where the nose starts
const NOSE_LEN = 180;
const SEG = 6;
const SEG_LEN = NOSE_LEN / SEG;
const SEG_CAP = 20; // each segment crushes by at most this
export const NOSE_CAP = SEG * SEG_CAP; // 120 units = 0.6 m
export const BARRIER_X = NOSE0 + NOSE_LEN + 8;
export const APPROACH = 160; // how far back the car starts its run
const WALL_Y = 60; // outer face of the cell wall
const WALL_T = 6;
const W_TOP = [108, 78, 66]; // wall heights: engine side, cockpit, footwell
const X_STEP = [0, 110, 236, 300];
const HALO_T = [224, 0, 158]; // apex
const HALO_M = 236; // front mount plate starts here
const HALO_RM = [126, 58, 84]; // rear mounts (mirrored in y)
const GANTRY_X = 224;
const GANTRY_Y = 104;
const RAM_X = 166;
const SIDE_FACE = WALL_Y + 30; // outer face of the side crash structure
export const RAM_REST = 136; // inner face of the ram pad before a run

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

export function projection() {
  const pts = [...k.boxCorners(BASE, -BASE_H - FOOT_H, 0), [GANTRY_X, 0, 262], [GANTRY_X, GANTRY_Y, 0], [BARRIER_X + 80, 0, 160], [-APPROACH, 0, 40]];
  return k.fitProjection(pts, WIDTH, HEIGHT, { pad: 26, azimuth: 72, elevation: 33 });
}

/* ---------- pose ---------- */

export function poseOf(s) {
  const pose = {
    carDx: s.carDx || 0, // sled shift along x, units
    crushU: s.crushU || 0, // nose crushed, units
    lean: s.lean || 0, // driver's forward pitch, rad
    skin: s.skin ?? 0.22, // how opaque the carbon skin is
    haloDefl: s.haloDefl || 0, // apex deflection, units (already magnified for the eye)
    engaged: s.engaged || 0, // load frame lowered onto the Halo, 0..1
    ramY: s.ramY ?? RAM_REST, // inner face of the ram pad
    panelC: s.panelC || 0, // side structure crushed, units
    intrU: s.intrU || 0, // cell wall pushed in, units
    cursor: s.cursor || 0, // time on the plot, s
    plotKey: s.plotKey || "",
    plot: s.plot || null,
    flash: s.flash || 0,
  };
  return pose;
}
export const REST_POSE = () => poseOf({});

/* ---------- helpers ---------- */

export function toolkit(P) {
  const S = (paths, style) => k.solidSvg(paths, style);
  const L = (d, style) => k.lineSvg(d, style);
  const D = (points, style) => k.dotsSvg(points, style);
  const at = (point) => k.iso(point, P);
  const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], a[2] ?? 0], P) - k.depthOf([b[0], b[1], b[2] ?? 0], P));
  return { S, L, D, at, byDepth };
}

const FY = G.frameOf([0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]); // prisms in the x-z plane, extruded along y
const axisX = (o) => G.frameOf(o, [1, 0, 0], [0, 1, 0], [0, 0, 1]);
const axisY = (o) => G.frameOf(o, [0, 1, 0], [1, 0, 0], [0, 0, 1]);

function bar(a, b, r, P, options = {}) {
  const length = dist3(a, b);
  const dir = [(b[0] - a[0]) / length, (b[1] - a[1]) / length, (b[2] - a[2]) / length];
  return G.disc(0, length, r, G.frameAlong(a, dir), P, options);
}

/** The Halo's three tubes, from its mounts to the apex, apex moved by the deflection. */
export function haloRoutes(defl) {
  const apex = add(HALO_T, [-0.5 * defl, 0, -0.85 * defl]);
  const mid = (sign) => add([172, sign * 46, 152], [-0.25 * defl, 0, -0.4 * defl]);
  return {
    apex,
    pillar: [[HALO_M + 16, 0, W_TOP[2] + 5], apex],
    arms: [1, -1].map((sign) => [[HALO_RM[0], sign * HALO_RM[1], HALO_RM[2]], mid(sign), apex]),
  };
}

/* ---------- items ---------- */

export function items(P) {
  const { S, L, D, at, byDepth } = toolkit(P);
  const V = G.viewOf(P);
  const list = [];
  const put = (item) => list.push(item);
  const tube = (route, r, tone, options = {}) => {
    const smooth = G.fillet(route, options.bend || r * 3, 8, { P, tube: r });
    return T.tubePieces(smooth, r, P, { maxLength: 16, spacing: 1.6, ...options.tube }).map((piece) => ({ key: G.dot3(piece.mid, V), svg: T.tubeSvg(piece, { tone }) }));
  };

  /* --- rig --- */
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
      // ruler strip along the front edge, and a label plate
      const strip = { x: 140, y: BASE.y + BASE.d - 22, w: 300, d: 9, r: 1 };
      out.push(S(k.slabOf(strip, 0, 0.7, P, 3), { tone: "lo", crease: "none" }));
      const t = k.topTicks(strip.x + 5, strip.x + strip.w - 5, 5, 5, strip.y + 1, 0.7, [2.6, 4.4], P);
      out.push(L(t.minor, { tone: "lo" }), L(t.major, { tone: "mid" }));
      out.push(D([at([strip.x + 2.5, strip.y + 4.5, 0.7]), at([strip.x + strip.w - 2.5, strip.y + 4.5, 0.7])], { size: 0.45 }));
      const label = { x: 470, y: BASE.y + 20, w: 90, d: 26, r: 2 };
      out.push(S(k.slabOf(label, 0, 0.9, P, 4), { tone: "lo", crease: "none" }));
      out.push(L(k.planOutline(k.insetPlan(label, 2.4), 0.9, P), { tone: "faint" }));
      [[8, 20, 72], [8, 13, 46], [8, 6, 58]].forEach(([dx, dy, len]) => out.push(L(k.lineOnTop([label.x + dx, label.y + dy], [label.x + dx + len, label.y + dy], 0.9, P), { tone: "lo", free: true })));
      out.push(D([at([label.x + 5, label.y + 5, 0.9]), at([label.x + label.w - 5, label.y + label.d - 5, 0.9])], { size: 0.45 }));
      return out.join("");
    })(),
  });
  put({
    id: "rails",
    group: "rig",
    svg: [-76, 76]
      .map((y) => {
        const rail = { x: -90, y: y - 6, w: 640, d: 12, r: 1.4 };
        return S(k.slabOf(rail, 0, 5, P, 3, 0.4), { tone: "mid" }) + D([-76, 0, 76, 152, 228, 304, 380, 456, 520].map((x) => at([x, y, 5])), { size: 0.45 });
      })
      .join(""),
  });

  /* --- the deceleration trace, drawn on the base --- */
  const PL = { x: -112, y: 90, w: 232, d: 80, r: 3 };
  const PX0 = PL.x + 14;
  const PX1 = PL.x + PL.w - 10;
  const PY0 = PL.y + PL.d - 12; // 0 g
  const T_MAX = 0.14;
  const G_SCALE = 0.5; // units per g
  const gy = (g) => PY0 - Math.min(g, 124) * G_SCALE;
  const tx = (t) => PX0 + (Math.min(t, T_MAX) / T_MAX) * (PX1 - PX0);
  put({
    id: "plot-plate",
    group: "plot",
    svg: (() => {
      const out = [];
      out.push(S(k.slabOf(PL, 0, 1.4, P, 4, 0.3), { tone: "lo", crease: "none" }));
      out.push(L(k.planOutline(k.insetPlan(PL, 4), 1.4, P, 3), { tone: "faint" }));
      // axes and the two limits: 60 g peak, 25 g average
      const frameR = PL.x + PL.w - 4; // the inset frame the axes and limits run into
      const frameT = PL.y + 4;
      out.push(L(k.lineOnTop([PX0, PY0], [frameR, PY0], 1.4, P) + k.lineOnTop([PX0, PY0], [PX0, frameT], 1.4, P), { tone: "mid" }));
      out.push(L(k.lineOnTop([PX0, gy(60)], [frameR, gy(60)], 1.4, P), { tone: "lo", dashed: true }));
      out.push(L(k.lineOnTop([PX0, gy(25)], [frameR, gy(25)], 1.4, P), { tone: "lo", dashed: true }));
      const ticks = k.topTicks(PX0, PX1, (PX1 - PX0) / 14, 7, PY0, 1.4, [2, 3.4], P);
      out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
      out.push(D(k.corners(PL, 4.5).map(([x, y]) => at([x, y, 1.4])), { size: 0.45 }));
      return out.join("");
    })(),
  });
  put({
    id: "plot-curve",
    deps: ["cursor", "plotKey"],
    group: "plot",
    live: (pose) => {
      const pl = pose.plot;
      if (!pl || !pl.trace.length) return "";
      const pts = [[PX0, PY0]];
      pl.trace.forEach((p) => pts.push([tx(p.t), gy(p.g)]));
      const last = pl.trace[pl.trace.length - 1];
      pts.push([tx(last.t + 0.002), PY0]);
      const full = k.onTop(pts, 1.5, P);
      let out = L(full, { tone: "mid" });
      const c = Math.min(pose.cursor, last.t);
      if (pose.cursor > 0) {
        const done = [[PX0, PY0]];
        pl.trace.forEach((p) => p.t <= c && done.push([tx(p.t), gy(p.g)]));
        const cur = pl.at(c);
        done.push([tx(c), gy(cur.g)]);
        out += L(k.onTop(done, 1.6, P), { tone: "lit" });
        out += L(k.lineOnTop([tx(c), PY0], [tx(c), gy(cur.g)], 1.6, P), { tone: "faint" });
        out += D([at([tx(c), gy(cur.g), 1.6])], { size: 1.1, tone: "hi" });
      }
      return out;
    },
  });

  /* --- the sled and its far side post --- */
  put({
    id: "gantry-far",
    group: "rig",
    svg: S(k.cylinder(GANTRY_X, -GANTRY_Y, 7, 0, 246, P, 24), { tone: "mid", crease: "none" }) + S(k.cylinder(GANTRY_X, -GANTRY_Y, 11, 0, 8, P, 24), { tone: "lo" }),
  });
  put({
    id: "sled",
    xform: "car",
    group: "rig",
    svg: (() => {
      const out = [];
      const plate = { x: -26, y: -92, w: 352, d: 184, r: 4 };
      out.push(S(k.slabOf(plate, 10, Z0 - 10, P, 4, 0.6), { tone: "mid" }));
      for (const y of [-76, 76]) for (const x of [-12, 36, 238, 282]) out.push(S(k.slabOf({ x: x - 0, y: y - 6, w: 36, d: 12, r: 1.4 }, 5, 5, P, 3, 0.3), { tone: "lo" }));
      out.push(L(k.planOutline(k.insetPlan(plate, 5), Z0, P, 4), { tone: "faint" }));
      out.push(D(k.corners(plate, 8).map(([x, y]) => at([x, y, Z0])), { size: 0.5 }));
      return out.join("");
    })(),
  });

  /* --- the tub: rear bulkhead, far wall, floor --- */
  const wall = (sign, withSeams) => {
    const out = [];
    [0, 1, 2].forEach((i) => {
      const x0 = X_STEP[i];
      const x1 = X_STEP[i + 1];
      const outer = sign * WALL_Y;
      const inner = sign * (WALL_Y - WALL_T);
      const taper = i === 2 ? 20 : 0;
      const poly = [
        [x0, outer],
        [x1, outer - sign * taper],
        [x1, inner - sign * taper],
        [x0, inner],
      ];
      const top = W_TOP[i];
      out.push(S(k.extrude(poly, Z0 + 6, top - Z0 - 6, P, { convex: true, bevel: 0.6 }), { tone: "mid" }));
      if (withSeams) {
        // composite ply seams along the outer face, end to end of the piece
        [0.3, 0.62].forEach((f) => {
          const z = Z0 + 6 + (top - Z0 - 6) * f;
          out.push(L(k.segment([x0, outer, z], [x1, outer - sign * taper, z], P), { tone: "faint" }));
        });
      }
    });
    return out.join("");
  };
  put({
    id: "tub-rear",
    cls: "fa-carbon",
    xform: "car",
    group: "shell",
    svg: (() => {
      const out = [];
      const plan = { x: 0, y: -WALL_Y, w: 10, d: WALL_Y * 2, r: 2 };
      out.push(S(k.slabOf(plan, Z0, W_TOP[0] - Z0, P, 3, 0.7), { tone: "mid" }));
      out.push(D([-44, -22, 0, 22, 44].flatMap((y) => [Z0 + 14, W_TOP[0] - 14].map((z) => at([10, y, z]))), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({ id: "tub-far", cls: "fa-carbon", xform: "car", group: "shell", svg: wall(-1, false) });
  put({
    id: "tub-floor",
    cls: "fa-carbon",
    xform: "car",
    group: "shell",
    svg: (() => {
      const out = [];
      const ringPts = [[0, -WALL_Y + WALL_T], [236, -WALL_Y + WALL_T], [300, -WALL_Y + WALL_T + 20], [300, WALL_Y - WALL_T - 20], [236, WALL_Y - WALL_T], [0, WALL_Y - WALL_T]];
      out.push(S(k.extrude(ringPts, Z0, 6, P, { convex: true }), { tone: "lo" }));
      // floor ribs, edge to edge
      for (const x of [60, 150, 236]) out.push(L(k.segment([x, -WALL_Y + WALL_T, Z0 + 6], [x, WALL_Y - WALL_T, Z0 + 6], P), { tone: "faint" }));
      return out.join("");
    })(),
  });

  /* --- inside the tub: fuel cell, seat, dummy, battery, electronics, legs, pedals --- */
  put({
    id: "fuel",
    xform: "car",
    group: "fuel",
    svg: (() => {
      const out = [];
      const tank = { x: 14, y: -48, w: 78, d: 96, r: 6 };
      out.push(S(k.slabOf(tank, Z0 + 6, 64, P, 6, 1.2), { tone: "hi" }));
      out.push(L(k.sideSeam(tank, Z0 + 6 + 22, P, 6) + k.sideSeam(tank, Z0 + 6 + 44, P, 6), { tone: "faint" }));
      out.push(L(k.planOutline(k.insetPlan(tank, 7), Z0 + 70, P, 5), { tone: "lo" }));
      // filler neck with a cap, a breather and a level sensor flange
      out.push(S(k.cylinder(60, 28, 6, Z0 + 70, 8, P, 24), { tone: "mid" }));
      out.push(S(k.cylinder(60, 28, 4, Z0 + 78, 3, P, 20), { tone: "hi", crease: "none" }));
      out.push(L(k.knurl(60, 28, 6, Z0 + 70, Z0 + 78, 18, P), { tone: "lo" }));
      out.push(S(k.cylinder(32, -26, 4, Z0 + 70, 5, P, 18), { tone: "mid" }));
      out.push(S(k.cylinder(32, 10, 5.5, Z0 + 70, 3, P, 20), { tone: "mid" }));
      out.push(D(k.corners(k.insetPlan(tank, 7), 0).map(([x, y]) => at([x, y, Z0 + 70])), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({
    id: "seat",
    xform: "car",
    group: "driver",
    svg: (() => {
      const out = [];
      out.push(S(k.slabOf({ x: 98, y: -26, w: 62, d: 52, r: 5 }, Z0 + 6, 12, P, 5, 0.7), { tone: "mid" }));
      out.push(S(G.prismOf(FY, [[98, 36], [92, 100], [106, 104], [112, 36]], -28, 28, P), { tone: "mid", crease: "faint" }));
      [-1, 1].forEach((s) => out.push(S(G.prismOf(FY, [[98, 36], [96, 70], [104, 74], [140, 38]], s > 0 ? 22 : -30, s > 0 ? 30 : -22, P), { tone: "lo" })));
      out.push(L(k.lineOnTop([112, -20], [150, -20], 38, P) + k.lineOnTop([112, 20], [150, 20], 38, P), { tone: "faint" }));
      return out.join("");
    })(),
  });
  put({
    id: "dummy",
    xform: "car",
    deps: ["lean"],
    group: "driver",
    live: (pose) => {
      const out = [];
      const hip = [128, 0, 46];
      const a = pose.lean;
      const rot = ([x, z]) => [hip[0] + x * Math.cos(a) + z * Math.sin(a), hip[2] - x * Math.sin(a) + z * Math.cos(a)];
      // torso: reclined slightly, pitching forward about the hip
      const torso = [[-6, 0], [14, 0], [8, 56], [-14, 56]].map(rot);
      out.push(S(G.prismOf(FY, torso, -24, 24, P), { tone: "hi", crease: "lo" }));
      const sh = rot([-2, 54]);
      const chest = rot([12, 30]);
      // harness: shoulder straps from the collar bone to an anchor behind the seat back, lap belt over the hips
      [-1, 1].forEach((s) => out.push(S(bar([chest[0], s * 12, chest[1]], [92, s * 12, 92], 1.6, P, { steps: 10 }), { tone: "lo", crease: "none" })));
      out.push(S(bar([hip[0] + 10, -24, hip[2] + 8], [hip[0] + 10, 24, hip[2] + 8], 2, P, { steps: 10 }), { tone: "lo", crease: "none" }));
      out.push(S(G.disc(0, 4, 4, axisX([chest[0] + 2, 0, chest[1] - 8]), P, { steps: 12 }), { tone: "hi", crease: "none" }));
      // neck and helmet
      const neckTop = rot([-6, 66]);
      out.push(S(G.disc(0, 12, 6, G.frameAlong([sh[0], 0, sh[1]], [neckTop[0] - sh[0], 0, neckTop[1] - sh[1]]), P, { steps: 14 }), { tone: "mid", crease: "none" }));
      const head = rot([-4, 80]);
      out.push(S(G.sphereOf([head[0], 0, head[1]], 17, P, { steps: 40 }), { tone: "hi", crease: "none" }));
      const visor = axisX([head[0] + 9, 0, head[1] + 2]);
      out.push(L(G.circleOf(visor, 8, 8, P), { tone: "lo" }));
      out.push(L(G.circleOf(axisY([head[0], 0, head[1]]), 17.1, 16, P), { tone: "faint" }));
      return out.join("");
    },
  });
  put({
    id: "battery",
    xform: "car",
    group: "electronics",
    svg: (() => {
      const out = [];
      const box = { x: 168, y: -40, w: 42, d: 80, r: 3 };
      out.push(S(k.slabOf(box, Z0 + 6, 20, P, 4, 0.9), { tone: "hi" }));
      out.push(L(k.sideSeam(box, Z0 + 6 + 12, P, 4), { tone: "faint" }));
      out.push(L(k.planOutline(k.insetPlan(box, 5), Z0 + 26, P, 3), { tone: "faint" }));
      // cell vents as closed slots, two terminals with caps
      for (const y of [-26, -12, 2, 16]) out.push(L(k.planOutline({ x: box.x + 12, y, w: 18, d: 6, r: 3 }, Z0 + 26, P, 4), { tone: "lo" }));
      out.push(S(k.cylinder(box.x + 8, -33, 3.4, Z0 + 26, 4, P, 16), { tone: "lit", lit: true }));
      out.push(S(k.cylinder(box.x + 8, 33, 3.4, Z0 + 26, 4, P, 16), { tone: "mid" }));
      return out.join("");
    })(),
  });
  put({
    id: "cables",
    xform: "car",
    group: "electronics",
    svg: (() => {
      const out = [];
      const routes = [
        [[176, -33, Z0 + 30], [176, -33, Z0 + 38], [236, -33, Z0 + 38], [250, -20, Z0 + 28]],
        [[176, 33, Z0 + 30], [176, 33, Z0 + 38], [232, 33, Z0 + 38], [250, 20, Z0 + 28]],
      ];
      const pieces = routes.flatMap((r) => tube(r, 1.4, "mid", { bend: 6 }));
      pieces.sort((a, b) => a.key - b.key).forEach((p) => out.push(p.svg));
      return out.join("");
    })(),
  });
  put({
    id: "ecu",
    xform: "car",
    group: "electronics",
    svg: (() => {
      const out = [];
      const box = { x: 214, y: -36, w: 74, d: 72, r: 3 };
      out.push(S(k.slabOf(box, Z0 + 6, 18, P, 4, 0.9), { tone: "hi" }));
      out.push(L(k.planOutline(k.insetPlan(box, 5), Z0 + 24, P, 3), { tone: "faint" }));
      // connector shells along the front edge and heat-sink fins on the lid
      for (const y of [-26, -10, 6, 22]) out.push(S(k.slabOf({ x: box.x + box.w - 3, y, w: 7, d: 10, r: 1.2 }, Z0 + 12, 8, P, 2, 0.2), { tone: "mid" }));
      for (let i = 0; i < 6; i++) out.push(L(k.planOutline({ x: box.x + 14 + i * 7, y: box.y + 10, w: 3, d: box.d - 20, r: 1.5 }, Z0 + 24, P, 3), { tone: "lo" }));
      out.push(D(k.corners(box, 5).map(([x, y]) => at([x, y, Z0 + 24])), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({
    id: "legs",
    xform: "car",
    group: "driver",
    svg: (() => {
      const out = [];
      for (const y of [-13, 13].sort((a, b) => a - b)) {
        const hipP = [132, y, 52];
        const knee = [214, y, 76];
        const ankle = [268, y, 56];
        out.push(S(bar(hipP, knee, 9.5, P, { steps: 20 }), { tone: "hi", crease: "none" }));
        out.push(S(G.sphereOf(knee, 10, P, { steps: 24 }), { tone: "hi", crease: "none" }));
        out.push(S(bar(knee, ankle, 7.5, P, { steps: 18 }), { tone: "hi", crease: "none" }));
        out.push(S(k.slabOf({ x: ankle[0] - 8, y: y - 8, w: 24, d: 16, r: 3 }, ankle[2] - 7, 9, P, 3, 0.5), { tone: "mid" }));
      }
      return out.join("");
    })(),
  });
  put({
    id: "pedals",
    xform: "car",
    group: "shell",
    svg: (() => {
      const out = [];
      out.push(S(k.slabOf({ x: 284, y: -34, w: 12, d: 68, r: 2 }, Z0 + 6, 56, P, 3, 0.6), { tone: "mid" }));
      [-18, 0, 18].forEach((y, i) => out.push(S(G.prismOf(FY, [[274, 40], [284, 40], [284, 76], [276, 76]], y - (i === 1 ? 7 : 5), y + (i === 1 ? 7 : 5), P), { tone: i === 1 ? "hi" : "mid", crease: "faint" })));
      out.push(D([0, 1, 2, 3].map((i) => at([296, -24 + i * 16, Z0 + 52])), { size: 0.5 }));
      return out.join("");
    })(),
  });

  /* --- skin: near wall, scuttle and front bulkhead fade away for the cutaway --- */
  put({ id: "tub-near", cls: "fa-carbon", xform: "car", skin: true, deps: ["skin"], group: "shell", svg: wall(1, true) });
  put({
    id: "scuttle",
    cls: "fa-carbon",
    xform: "car",
    skin: true,
    deps: ["skin"],
    group: "shell",
    svg: (() => {
      const out = [];
      const ringPts = [[236, -WALL_Y + WALL_T], [300, -WALL_Y + WALL_T + 20], [300, WALL_Y - WALL_T - 20], [236, WALL_Y - WALL_T]];
      out.push(S(k.extrude(ringPts, W_TOP[2], 5, P, { convex: true, bevel: 0.5 }), { tone: "mid" }));
      out.push(D([0.25, 0.5, 0.75].map((f) => at([252 + f * 40, 0, W_TOP[2] + 5])), { size: 0.5 }));
      out.push(S(k.slabOf({ x: 294, y: -WALL_Y + WALL_T + 20, w: 12, d: (WALL_Y - WALL_T - 20) * 2, r: 2 }, Z0 + 6, W_TOP[2] - Z0 - 6 + 5, P, 3, 0.7), { tone: "hi" }));
      return out.join("");
    })(),
  });

  /* --- the Halo: three mounts, then the titanium tubes --- */
  put({
    id: "halo-mounts",
    xform: "car",
    group: "halo",
    svg: (() => {
      const out = [];
      out.push(S(k.slabOf({ x: HALO_M + 8, y: -14, w: 32, d: 28, r: 3 }, W_TOP[2] + 5, 5, P, 3, 0.5), { tone: "hi" }));
      out.push(D([[HALO_M + 14, -8], [HALO_M + 14, 8], [HALO_M + 34, -8], [HALO_M + 34, 8]].map(([x, y]) => at([x, y, W_TOP[2] + 10])), { size: 0.55 }));
      for (const sgn of [-1, 1]) {
        out.push(S(k.slabOf({ x: HALO_RM[0] - 16, y: sgn * HALO_RM[1] - 8, w: 32, d: 16, r: 2.4 }, W_TOP[1], 5, P, 3, 0.4), { tone: "hi" }));
        out.push(D([[-9, -4], [-9, 4], [9, -4], [9, 4]].map(([dx, dy]) => at([HALO_RM[0] + dx, sgn * HALO_RM[1] + dy, W_TOP[1] + 5])), { size: 0.5 }));
      }
      return out.join("");
    })(),
  });
  put({
    id: "halo",
    xform: "car",
    deps: ["haloDefl"],
    group: "halo",
    live: (pose) => {
      const r = haloRoutes(pose.haloDefl);
      const pieces = [...tube(r.pillar, 5.4, "hi", { bend: 8 }), ...r.arms.flatMap((route) => tube(route, 4.8, "hi", { bend: 24 }))];
      pieces.sort((a, b) => a.key - b.key);
      return pieces.map((p) => p.svg).join("") + S(G.sphereOf(r.apex, 7.2, P, { steps: 28 }), { tone: "hi", crease: "none" }) + [1, -1].map((s) => S(G.sphereOf([HALO_RM[0], s * HALO_RM[1], HALO_RM[2]], 6, P, { steps: 20 }), { tone: "hi", crease: "none" })).join("");
    },
  });

  /* --- side impact: crush panel and ram --- */
  put({
    id: "side-panel",
    xform: "car",
    deps: ["panelC", "intrU"],
    group: "side",
    live: (pose) => {
      const out = [];
      const thick = Math.max(2, 30 - pose.panelC);
      const y0 = WALL_Y;
      const plan = { x: 122, y: y0, w: 90, d: thick, r: 1.6 };
      out.push(S(k.slabOf(plan, Z0 + 14, 40, P, 3, 0.4), { tone: "mid" }));
      // honeycomb cells on the top face: rows of closed slots that close up as the panel crushes
      const cells = [];
      for (let i = 0; i < 6; i++) cells.push(k.planOutline({ x: plan.x + 6 + i * 14, y: y0 + 3, w: 9, d: Math.max(1.2, thick - 6), r: 1.4 }, Z0 + 54, P, 3));
      out.push(L(cells.join(""), { tone: "lo" }));
      return out.join("");
    },
  });
  put({
    id: "ram",
    deps: ["ramY"],
    group: "side",
    live: (pose) => {
      const out = [];
      const y = pose.ramY;
      // carriage on a rail near the base edge, rod, pad
      out.push(S(k.slabOf({ x: RAM_X - 8, y: 100, w: 16, d: 70, r: 1.4 }, 0, 4, P, 3, 0.3), { tone: "lo" }));
      out.push(S(k.slabOf({ x: RAM_X - 26, y: 146, w: 52, d: 24, r: 3 }, 4, 38, P, 4, 0.7), { tone: "mid" }));
      out.push(S(G.disc(0, Math.max(1, 148 - y), 7, axisY([RAM_X, y, Z0 + 34]), P, { steps: 22 }), { tone: "lit", crease: "none", lit: true }).replace("transform=", "transform="));
      out.push(S(k.slabOf({ x: RAM_X - 34, y, w: 68, d: 10, r: 2 }, Z0 + 10, 50, P, 3, 0.6), { tone: "hi" }));
      out.push(L(k.planOutline(k.insetPlan({ x: RAM_X - 34, y, w: 68, d: 10, r: 2 }, 2), Z0 + 60, P, 3), { tone: "faint" }));
      return out.join("");
    },
  });

  /* --- the nose: six crush segments, front ones go first --- */
  put({
    id: "nose",
    cls: "fa-carbon",
    xform: "car",
    deps: ["crushU"],
    group: "crash",
    live: (pose) => {
      const out = [];
      // how much each segment has shortened, front first
      let left = pose.crushU;
      const crushed = Array.from({ length: SEG }, () => 0);
      for (let i = SEG - 1; i >= 0 && left > 0; i--) {
        crushed[i] = Math.min(SEG_CAP, left);
        left -= crushed[i];
      }
      let x = NOSE0;
      const zb = Z0 + 8;
      const ztop = (xx) => Z0 + 8 + 56 - 0.2 * (xx - NOSE0);
      const half = (xx) => 34 - 0.085 * (xx - NOSE0);
      const bodies = [];
      for (let i = 0; i < SEG; i++) {
        const nominal0 = NOSE0 + i * SEG_LEN;
        const nominal1 = nominal0 + SEG_LEN;
        const len = SEG_LEN - crushed[i];
        const f = crushed[i] / SEG_CAP; // fold-up of this segment
        const x0 = x;
        const x1 = x + len;
        const hw = ((half(nominal0) + half(nominal1)) / 2) * (1 + 0.28 * f);
        const grow = 1 + 0.3 * f;
        const poly = [[x0, zb], [x1 - 0.4, zb], [x1 - 0.4, zb + (ztop(nominal1) - zb) * grow], [x0, zb + (ztop(nominal0) - zb) * grow]];
        bodies.push(S(G.prismOf(FY, poly, -hw, hw, P), { tone: i % 2 ? "mid" : "hi", crease: "faint" }));
        // a rib line on the side where the segments meet, end to end of the face
        x = x1;
      }
      out.push(...bodies);
      // front cap and the mounting plate at the bulkhead
      const tipX = x;
      out.push(S(G.prismOf(FY, [[tipX, zb], [tipX + 6, zb + 2], [tipX + 6, zb + 18], [tipX, ztop(NOSE0 + NOSE_LEN) + 2]], -18, 18, P), { tone: "lit", lit: true, crease: "faint" }));
      return out.join("");
    },
  });
  put({
    id: "nose-plate",
    xform: "car",
    group: "crash",
    svg: (() => {
      const out = [];
      out.push(S(k.slabOf({ x: NOSE0 - 6, y: -34, w: 6, d: 68, r: 2 }, Z0 + 6, 62, P, 3, 0.5), { tone: "hi" }));
      out.push(D([-24, -8, 8, 24].flatMap((y) => [Z0 + 16, Z0 + 56].map((z) => at([NOSE0, y, z]))), { size: 0.5 }));
      return out.join("");
    })(),
  });

  /* --- the wall, and the near gantry with the Halo ram --- */
  put({
    id: "barrier",
    group: "rig",
    svg: (() => {
      const out = [];
      const w = { x: BARRIER_X, y: -130, w: 84, d: 260, r: 4 };
      out.push(S(k.slabOf({ x: w.x - 6, y: w.y - 12, w: w.w + 12, d: w.d + 24, r: 4 }, 0, 8, P, 4, 0.6), { tone: "lo" }));
      out.push(S(k.slabOf(w, 8, 150, P, 4, 1), { tone: "mid" }));
      out.push(L(k.sideSeam(w, 60, P, 4) + k.sideSeam(w, 110, P, 4), { tone: "faint" }));
      out.push(D(k.dotGrid(w.x + 14, w.y + 20, 5, 12, 16, 158, P), { size: 0.5, tone: "lo" }));
      // four load cells on the wall face are on the far side; bolts along the top edge here
      out.push(D(k.corners(w, 8).map(([x, y]) => at([x, y, 158])), { size: 0.6 }));
      return out.join("");
    })(),
  });
  put({
    id: "gantry-near",
    deps: ["haloDefl", "engaged", "carDx"],
    group: "halo",
    live: (pose) => {
      const out = [];
      const r = haloRoutes(pose.haloDefl);
      const apex = [GANTRY_X, 0, r.apex[2]];
      const touch = apex[2] + 8;
      const padZ = touch + (1 - pose.engaged) * 58;
      out.push(S(k.cylinder(GANTRY_X, GANTRY_Y, 7, 0, 246, P, 24), { tone: "mid", crease: "none" }) + S(k.cylinder(GANTRY_X, GANTRY_Y, 11, 0, 8, P, 24), { tone: "lo" }));
      const beam = { x: GANTRY_X - 14, y: -GANTRY_Y - 12, w: 28, d: GANTRY_Y * 2 + 24, r: 3 };
      out.push(S(k.slabOf(beam, 246, 14, P, 3, 0.6), { tone: "mid" }));
      out.push(D([-GANTRY_Y, GANTRY_Y].map((y) => at([GANTRY_X, y, 260])), { size: 0.6 }));
      // the ram: body under the beam, rod down to a pad that sits on the apex
      out.push(S(k.cylinder(apex[0], 0, 12, 214, 32, P, 32, 0.6), { tone: "hi" }));
      out.push(L(k.sideArc(apex[0], 0, 12, 224, P, 20) + k.sideArc(apex[0], 0, 12, 236, P, 20), { tone: "faint" }));
      out.push(S(k.cylinder(apex[0], 0, 6, padZ + 8, Math.max(1, 214 - padZ - 8), P, 20), { tone: "lit", crease: "none", lit: true }));
      out.push(S(k.cylinder(apex[0], 0, 11, padZ, 8, P, 28, 0.4), { tone: "hi" }));
      return out.join("");
    },
  });
  return list;
}

export const markup = (item, pose) => (item.live ? item.live(pose) : item.svg || "");

export function shiftOf(item, pose) {
  if (item.xform === "car") return [pose.carDx, 0, 0];
  return [0, 0, 0];
}

// Geometry of the wind-tunnel figure: a simplified F1 car on a rolling road, with smoke-rake streamlines.
// 1 unit = 1 cm. x runs forward (the nose is at +x), y to the car's left, z up. The belt top is z = 0.
// Runs in Node (the build draws the page) and in the browser (the live page redraws what moves).
import * as k from "../../vendor/kit/iso-kit.mjs";
import * as G from "../../vendor/kit/lathe.mjs";

export const WIDTH = 960;
export const HEIGHT = 580;
export const H0 = 3.5; // plank height above the road at build time, in units (35 mm)

const PLATFORM = { x: -345, y: -152, w: 715, d: 304, r: 12 };
const PLAT_H = 14;
const FOOT_H = 6;
const BELT = { x: -322, y: -110, w: 636, d: 220, r: 6 };

const WHEEL_F = { x: 190, y: 88, r: 33, w: 15 };
const WHEEL_R = { x: -175, y: 90, r: 34, w: 20 };

export function projection() {
  const pts = [
    ...k.boxCorners(PLATFORM, -PLAT_H - FOOT_H, 0),
    [-262, 0, 110],
    [332, 0, 60],
    [60, 0, 110],
  ];
  return k.fitProjection(pts, WIDTH, HEIGHT, { pad: 22, azimuth: 38, elevation: 28 });
}

/* ---------- pose ---------- */

/** What the drawing needs: ride in mm, flap angles in degrees, drs 0..1, forces in kN, xray 0..1, spin. */
export function poseOf({ ride = 35, fw = 15, rw = 22, drs = 0, xray = 0, spin = 0, belt = 0, f = {}, hot = 0, dial = 0 }) {
  return {
    ride, fw, rw, drs, xray, spin, belt, f, hot, dial,
    h: ride / 10, // units
    carShift: ride / 10 - H0,
    key: [ride.toFixed(1), fw.toFixed(1), rw.toFixed(1), drs.toFixed(2)].join("|"),
    fkey: f && f.down ? Math.round(f.down / 150) + "|" + Math.round(f.drag / 150) + "|" + drs.toFixed(2) : "0",
  };
}

export const REST_POSE = () => poseOf({});

/* ---------- helpers ---------- */

export function toolkit(P) {
  const S = (paths, style) => k.solidSvg(paths, style);
  const L = (d, style) => k.lineSvg(d, style);
  const D = (points, style) => k.dotsSvg(points, style);
  const at = (point) => k.iso(point, P);
  const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
  const poly3 = (points, closed = true) => k.pathOf(points.map((p) => k.iso(p, P)), closed);
  return { S, L, D, at, byDepth, poly3 };
}

const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const axisY = (o) => G.frameOf(o, [0, 1, 0], [1, 0, 0], [0, 0, 1]);
const axisX = (o) => G.frameOf(o, [1, 0, 0], [0, 1, 0], [0, 0, 1]);

/** A straight round bar between two world points, as a solid. */
function bar(a, b, r, P, options = {}) {
  const length = dist3(a, b);
  const dir = [(b[0] - a[0]) / length, (b[1] - a[1]) / length, (b[2] - a[2]) / length];
  return G.disc(0, length, r, G.frameAlong(a, dir), P, options);
}

/**
 * An aerofoil section as a convex polygon in (x, z): the leading edge is at +x, the curved
 * (suction) side faces down like an inverted wing, and the trailing edge is raised by `deg`.
 * `pivot` is where it turns, as a fraction of the chord from the trailing edge.
 */
export function foil(cx, cz, chord, thick, deg, pivot = 0.5) {
  const pts = [];
  const n = 9;
  const top = [];
  const bottom = [];
  for (let i = 0; i <= n; i++) {
    const s = i / n;
    const u = chord / 2 - s * chord;
    const shape = Math.sin(Math.PI * Math.pow(s, 0.62));
    top.push([u, thick * 0.26 * shape]);
    bottom.push([u, -thick * 0.74 * shape * (1 - 0.18 * s)]);
  }
  pts.push(...top, ...bottom.reverse().slice(1, -1));
  const phi = (-deg * Math.PI) / 180;
  const px = chord / 2 - pivot * chord;
  return pts.map(([u, v]) => {
    const du = u - px;
    return [cx + px + du * Math.cos(phi) - v * Math.sin(phi), cz + du * Math.sin(phi) + v * Math.cos(phi)];
  });
}

/** An extruded aerofoil across the span, from y0 to y1. */
function wing(polygon, y0, y1, P, style) {
  const F = G.frameOf([0, y0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
  return k.solidSvg(G.prismOf(F, polygon, 0, y1 - y0, P), style);
}

/* ---------- the shape of the car, for the airflow ---------- */

/** Height of the top of the car at (x, y) above the road, in units, for the streamlines to clear. */
export function topProfile(x, y, h) {
  const ay = Math.abs(y);
  let z = 0;
  const zf = h + 4.1;
  const inX = (a, b) => x >= a && x <= b;
  if (ay < 70 && inX(-185, 165)) z = Math.max(z, zf + 1);
  if (ay < 20) {
    if (inX(150, 258)) z = Math.max(z, zf + 14 + (x - 150) * -0.05);
    if (inX(-35, 150)) z = Math.max(z, zf + 44);
    if (inX(5, 95)) z = Math.max(z, zf + 70);
    if (inX(-105, -25)) z = Math.max(z, zf + 96 - Math.abs(x + 65) * 0.2);
    if (inX(-215, -100)) z = Math.max(z, zf + 54);
    if (inX(-262, -212)) z = Math.max(z, zf + 82);
  }
  if (ay >= 20 && ay < 58 && inX(-175, 30)) z = Math.max(z, zf + 40);
  if (ay < 100 && inX(205, 262)) z = Math.max(z, zf + 20);
  const wheel = (w) => {
    if (ay > w.y - w.w - 2 && ay < w.y + w.w + 2 && Math.abs(x - w.x) < w.r) z = Math.max(z, w.r + Math.sqrt(Math.max(0, w.r * w.r - (x - w.x) ** 2)));
  };
  wheel(WHEEL_F);
  wheel(WHEEL_R);
  return z;
}

const smoothstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** A streamline above the car at lateral position y. start is the height at the smoke rake. */
function upperLine(y, start, pose, kind) {
  const xs = [];
  for (let x = 322; x >= -322; x -= 6) xs.push(x);
  const raw = xs.map((x) => topProfile(x, y, pose.h));
  const win = 7;
  const lifted = raw.map((_, i) => {
    let m = 0;
    for (let j = Math.max(0, i - win); j <= Math.min(raw.length - 1, i + win); j++) m = Math.max(m, raw[j]);
    return m;
  });
  const smooth = lifted.map((_, i) => {
    let sum = 0;
    let n = 0;
    for (let j = Math.max(0, i - 4); j <= Math.min(lifted.length - 1, i + 4); j++) {
      sum += lifted[j];
      n++;
    }
    return sum / n;
  });
  const clear = 7 + (Math.abs(y) % 5);
  const pts = xs.map((x, i) => {
    let z = smooth[i] + clear;
    // the front wing pushes air up in front of it (upwash) and the rear wing pulls it down behind it
    z += 9 * Math.exp(-((x - 268) ** 2) / 1200);
    if (Math.abs(y) < 30) z += 5 * Math.exp(-((x + 262) ** 2) / 900) * (1 - 0.5 * pose.drs);
    const blend = smoothstep(275, 322, x);
    z = z * (1 - blend) + start * blend;
    if (x < -262) z -= (-262 - x) * (kind === "wake" ? 0.11 : 0.05); // downwash behind the wing
    return [x, y, Math.max(z, 3)];
  });
  return pts;
}

/** The streamline that runs between the rear wing's main plane and flap: it only gets through when DRS opens the slot. */
function slotLine(y, pose) {
  const xs = [];
  for (let x = 322; x >= -322; x -= 6) xs.push(x);
  return xs.map((x) => {
    const base = upperLine(y, 36, pose, "").find((p) => p[0] === x)[2];
    const through = 91 - Math.max(0, -240 - x) * 0.04;
    const over = 115 + Math.max(0, -250 - x) * -0.02;
    const near = smoothstep(-205, -232, x) * (1 - smoothstep(-262, -290, x));
    const over2 = base + (over - base) * near;
    const thru = base + (through - base) * near;
    return [x, y, over2 + (thru - over2) * pose.drs];
  });
}

/** Underbody streamline at lateral position y: it runs in the gap between plank and road, fast through the throat. */
function underLine(y, frac, pose) {
  const xs = [];
  for (let x = 322; x >= -322; x -= 6) xs.push(x);
  const h = pose.h;
  return xs.map((x) => {
    // gap between road and floor: wide at the inlet, narrowest at the throat, opening again in the diffuser
    let gap = h + 1.6;
    if (x > 40) gap = h + 1.6 + 7 * smoothstep(40, 150, x);
    if (x < -120) gap += (-120 - x) * 0.19;
    if (x < -256) gap = h + 1.6 + (256 - 120) * 0.19;
    if (x > 168) gap = Math.max(gap, h + 8);
    const f = x > 200 ? frac * 0.7 + 0.2 : frac;
    return [x, y, Math.max(0.4, gap * (x > 322 - 40 ? 1 : f))];
  });
}

/* ---------- items ---------- */

/**
 * Items back to front. Fields: id, group, xform ("car" moves with the ride height), cls (class on
 * the group), deps (pose fields a live item depends on), svg (static) or live(pose).
 */
export function items(P) {
  const { S, L, D, at, byDepth, poly3 } = toolkit(P);
  const list = [];
  const put = (item) => list.push(item);
  const flowStyle = (tone, extra = "") => `class="iso-line fa-flow" data-tone="${tone}" data-free="" style="stroke-dasharray:6 6;stroke-width:0.95${extra}"`;
  const lineTo = (pts, tone, extra = "") => `<path ${flowStyle(tone, extra)} d="${k.pathOf(pts.map((p) => k.iso(p, P)))}"/>`;

  /* --- tunnel platform, belt, rollers --- */
  put({
    id: "platform",
    group: "rig",
    svg: (() => {
      const out = [`<path class="iso-halo" d="${k.haloOf(PLATFORM, -PLAT_H - FOOT_H, PLAT_H, P)}"/>`];
      for (const [x, y] of byDepth(k.corners(PLATFORM, 24))) {
        out.push(S(k.cylinder(x, y, 9, -PLAT_H - FOOT_H, 2, P, 28), { tone: "lo" }));
        out.push(S(k.cylinder(x, y, 7, -PLAT_H - FOOT_H + 2, FOOT_H - 2, P, 28), { tone: "mid" }));
      }
      out.push(S(k.slabOf(PLATFORM, -PLAT_H, PLAT_H - 1.6, P, 8, 1.6), { tone: "mid" }));
      out.push(L(k.planOutline(k.insetPlan(PLATFORM, 7), -1.6, P), { tone: "faint" }));
      const screws = k.corners(PLATFORM, 13);
      out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.6, -1.6, P, 16)).join(""), { tone: "lo" }));
      out.push(D(screws.map(([x, y]) => at([x, y, -1.6])), { size: 0.5 }));
      // ruler along the front edge
      const strip = { x: -300, y: PLATFORM.y + PLATFORM.d - 22, w: 240, d: 9, r: 1 };
      out.push(S(k.slabOf(strip, -1.6, 0.7, P, 3), { tone: "lo", crease: "none" }));
      const ticks = k.topTicks(strip.x + 5, strip.x + strip.w - 5, 5, 5, strip.y + 1, -0.9, [2.6, 4.4], P);
      out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
      out.push(D([at([strip.x + 2.5, strip.y + 4.5, -0.9]), at([strip.x + strip.w - 2.5, strip.y + 4.5, -0.9])], { size: 0.45 }));
      const label = { x: -320, y: PLATFORM.y + 14, w: 90, d: 24, r: 2 };
      out.push(S(k.slabOf(label, -1.6, 0.9, P, 4), { tone: "lo", crease: "none" }));
      out.push(L(k.planOutline(k.insetPlan(label, 2.4), -0.7, P), { tone: "faint" }));
      [[8, 20, 70], [8, 13, 46], [8, 6, 58]].forEach(([dx, dy, len]) => out.push(L(k.lineOnTop([label.x + dx, label.y + dy], [label.x + dx + len, label.y + dy], -0.7, P), { tone: "lo", free: true })));
      out.push(D([at([label.x + 5, label.y + 5, -0.7]), at([label.x + label.w - 5, label.y + label.d - 5, -0.7])], { size: 0.45 }));
      return out.join("");
    })(),
  });
  put({
    id: "belt",
    group: "rig",
    svg: (() => {
      const out = [];
      out.push(S(k.slabOf(BELT, -1.6, 1.6, P, 6, 0.4), { tone: "lo" }));
      out.push(L(k.planOutline(k.insetPlan(BELT, 5), 0, P, 6), { tone: "faint" }));
      // belt guide rails along both sides and rollers at both ends
      [BELT.y - 7, BELT.y + BELT.d + 1].forEach((y) => out.push(S(k.slabOf({ x: BELT.x - 4, y, w: BELT.w + 8, d: 6, r: 1.4 }, -1.6, 4.2, P, 3, 0.5), { tone: "mid" })));
      [BELT.x - 10, BELT.x + BELT.w + 6].forEach((x) => {
        out.push(S(G.disc(-BELT.d / 2 - 7, BELT.d / 2 + 7, 5.4, axisY([x + 2, 0, -1]), P, { steps: 24 }), { tone: "mid" }));
      });
      out.push(D([at([BELT.x + 6, BELT.y - 4, 2.6]), at([BELT.x + BELT.w - 6, BELT.y - 4, 2.6]), at([BELT.x + 6, BELT.y + BELT.d + 4, 2.6]), at([BELT.x + BELT.w - 6, BELT.y + BELT.d + 4, 2.6])], { size: 0.45 }));
      return out.join("");
    })(),
  });
  put({
    id: "belt-stripes",
    group: "rig",
    deps: ["belt"],
    live: (pose) => {
      // the belt carries marks that run backwards at road speed, as if the car were driving forwards
      const pitch = 36;
      const out = [];
      for (let x = BELT.x + 10 - (pose.belt % pitch); x < BELT.x + BELT.w - 8; x += pitch) {
        if (x < BELT.x + 12) continue;
        out.push(`M${k.iso([x, BELT.y + 14, 0.05], P).join(" ")}L${k.iso([x, BELT.y + 28, 0.05], P).join(" ")}`);
        out.push(`M${k.iso([x, BELT.y + BELT.d - 28, 0.05], P).join(" ")}L${k.iso([x, BELT.y + BELT.d - 14, 0.05], P).join(" ")}`);
      }
      return L(out.join(""), { tone: "mid", free: true });
    },
  });

  /* --- far wheels, far suspension --- */
  const wheelParts = (w, side) => {
    const yc = side * w.y;
    const F = axisY([w.x, yc, w.r]);
    const prof = [[-w.w + 1, w.r - 11], [-w.w - 0.5, w.r - 7, 1], [-w.w - 0.5, w.r - 3, 1], [-w.w + 2.4, w.r - 0.6, 1], [-w.w + 6, w.r], [w.w - 6, w.r], [w.w - 2.4, w.r - 0.6, 1], [w.w + 0.5, w.r - 3, 1], [w.w + 0.5, w.r - 7, 1], [w.w - 1, w.r - 11]];
    return { F, prof, yc };
  };
  const wheelStatic = (w, side) => {
    const { F, prof } = wheelParts(w, side);
    const out = [];
    out.push(S(G.lathe(prof, F, P, { smooth: true }), { tone: "hi", crease: "faint" }));
    const face = side > 0 ? w.w + 0.5 : -w.w - 0.5;
    out.push(L(G.circleOf(F, face, w.r - 5.5, P) + G.circleOf(F, face, w.r - 11, P), { tone: "faint" }));
    return out.join("");
  };
  const wheelFace = (w, side, spin) => {
    const { F } = wheelParts(w, side);
    const face = side > 0 ? w.w + 0.6 : -w.w - 0.6;
    const out = [];
    out.push(`<path d="${G.circleOf(F, face, w.r - 11.5, P)}" style="fill:var(--anatomy-shade-0);stroke:none"/>`);
    out.push(L(G.spokesOf(F, face, 5, w.r - 12, 10, P, { phase: -spin }), { tone: "mid" }));
    out.push(S(G.disc(face - 0.3, face + 1.6 * (side > 0 ? 1 : -1), 4.4, F, P, { steps: 18 }), { tone: "hi", crease: "none" }));
    out.push(L(G.circleOf(F, face, w.r - 12, P), { tone: "lo" }));
    return out.join("");
  };
  const arms = (w, side, x0, y0, zs) => {
    // two wishbone arms and a rod from the chassis to the wheel hub
    const out = [];
    const hub = [w.x, side * (w.y - w.w - 2), w.r];
    zs.forEach(([z0, zh], i) => out.push(S(bar([x0 + (i ? 14 : -10), side * y0, z0], [hub[0] + (i ? 3 : -3), hub[1], hub[2] + zh], 1.3, P, { steps: 14 }), { tone: "mid", crease: "none" })));
    return out.join("");
  };
  const frontArms = (side) => arms(WHEEL_F, side, 120, 20, [[12, 11], [26, -12], [20, 0]].slice(0, 2));
  const rearArms = (side) => arms(WHEEL_R, side, -178, 18, [[14, 12], [24, -10]]);

  put({ id: "wheel-rl", group: "wheels", cls: "fa-rubber", svg: wheelStatic(WHEEL_R, -1) });
  put({ id: "arms-rl", group: "wheels", svg: rearArms(-1) });
  put({ id: "wheel-fl", group: "wheels", cls: "fa-rubber", svg: wheelStatic(WHEEL_F, -1) });
  put({ id: "arms-fl", group: "wheels", svg: frontArms(-1) });

  /* --- floor group: plank, floor plate, diffuser --- */
  put({
    id: "floor",
    group: "floor",
    xform: "car",
    cls: "fa-floor",
    deps: ["xray"],
    svg: (() => {
      const out = [];
      const zp = H0;
      // plank, floor plate (convex octagon), edge fences
      out.push(S(k.slabOf({ x: -125, y: -34, w: 230, d: 68, r: 3 }, zp, 1.6, P, 3), { tone: "lo", crease: "none" }));
      const ring = [[168, -48], [168, 48], [128, 68], [-120, 68], [-176, 40], [-176, -40], [-120, -68], [128, -68]];
      out.push(S(k.extrude(ring, zp + 1.6, 2.5, P, { bevel: 0.5 }), { tone: "mid" }));
      out.push(L(k.onTop(ring.map(([x, y]) => [x * 0.94 - Math.sign(x) * 0, y * 0.9]), zp + 4.1, P, true), { tone: "faint" }));
      // floor edge fences along both sides
      [-1, 1].forEach((side) => {
        out.push(S(k.slabOf({ x: -105, y: side * 66 - 0.9, w: 160, d: 1.8, r: 0.6 }, zp + 4.1, 7.5, P, 2, 0.3), { tone: "mid", crease: "faint" }));
        out.push(S(k.slabOf({ x: -60, y: side * 60 - 0.9, w: 70, d: 1.8, r: 0.6 }, zp + 4.1, 4, P, 2), { tone: "mid", crease: "faint" }));
      });
      out.push(D(k.corners({ x: -120, y: -64, w: 240, d: 128 }, 6).map(([x, y]) => at([x, y, zp + 4.1])), { size: 0.45 }));
      return out.join("");
    })(),
  });
  put({
    id: "diffuser",
    group: "diffuser",
    xform: "car",
    cls: "fa-floor",
    svg: (() => {
      const out = [];
      const zp = H0;
      // the diffuser is a ramp: its side profile shows the expansion of the flow under the car
      const ramp = [[-126, zp + 1.6], [-258, zp + 1.6 + 26], [-258, zp + 1.6 + 29.5], [-126, zp + 4.1]];
      const F = G.frameOf([0, -46, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
      out.push(k.solidSvg(G.prismOf(F, ramp, 0, 92, P), { tone: "hi", crease: "lo" }));
      // two strakes and the beam wing's lower mount on the ramp's upper face
      [-15, 15].forEach((y) => out.push(S(k.slabOf({ x: -250, y: y - 0.7, w: 106, d: 1.4, r: 0.5 }, zp + 3.3, 0.1, P, 2), { tone: "lo", crease: "none", flat: true })));
      return out.join("");
    })(),
  });
  put({
    id: "lines-under",
    group: "flow",
    cls: "fa-xray",
    deps: ["key"],
    live: (pose) =>
      [-52, -24, 0, 24, 52]
        .map((y, i) => {
          const pts = underLine(y, 0.3 + 0.12 * (i % 3), pose);
          // through the throat the air is fastest: a thicker, tighter dash that runs twice as quickly
          const fast = pts.filter((p) => p[0] <= 60 && p[0] >= -140);
          return lineTo(pts, "blue") + lineTo(fast, "blue", ";stroke-dasharray:3 4;stroke-width:2").replace("fa-flow", "fa-flow fa-fast");
        })
        .join(""),
  });
  put({
    id: "lines-far",
    group: "flow",
    deps: ["key"],
    live: (pose) => [-118, -74].map((y, i) => lineTo(upperLine(y, 14 + 12 * i, pose, ""), "mid")).join("") + lineTo(upperLine(-40, 30, pose, ""), "mid"),
  });

  /* --- rear wing and rear structure, behind the engine --- */
  const RW_SPAN = 51;
  put({
    id: "rw-end-far",
    group: "rearwing",
    xform: "car",
    svg: S(k.slabOf({ x: -266, y: -RW_SPAN - 2.6, w: 54, d: 2.6, r: 1 }, H0 + 50, 52, P, 3, 0.5), { tone: "mid" }) + L(k.segment([-248, -RW_SPAN - 2.6, H0 + 50], [-248, -RW_SPAN - 2.6, H0 + 102], P), { tone: "faint" }),
  });
  put({
    id: "rw-planes",
    group: "rearwing",
    xform: "car",
    deps: ["rw", "drs"],
    live: (pose) => {
      const out = [];
      const h = H0;
      // lower beam wing, main plane, then the upper flap that DRS opens
      out.push(wing(foil(-246, h + 61, 17, 3.6, 8), -RW_SPAN, RW_SPAN, P, { tone: "mid" }));
      // the flap pivots about a hinge near its trailing edge: opening lifts its leading edge, so incidence falls
      const flapDeg = pose.rw + 6 - pose.drs * 21;
      out.push(wing(foil(-249, h + 94, 21, 4.4, flapDeg, 0.1), -RW_SPAN, RW_SPAN, P, { tone: "lit", lit: true }));
      out.push(wing(foil(-237, h + 80, 25, 5.2, pose.rw + 2), -RW_SPAN, RW_SPAN, P, { tone: "hi", crease: "lo" }));
      return out.join("");
    },
  });
  put({
    id: "rw-pylon",
    group: "rearwing",
    xform: "car",
    svg: S(k.slabOf({ x: -243, y: -2.6, w: 11, d: 5.2, r: 1.2 }, H0 + 22, 56, P, 3, 0.4), { tone: "mid" }),
  });
  put({
    id: "rw-end-near",
    group: "rearwing",
    xform: "car",
    svg: S(k.slabOf({ x: -266, y: RW_SPAN, w: 54, d: 2.6, r: 1 }, H0 + 50, 52, P, 3, 0.5), { tone: "mid" }) + L(k.segment([-248, RW_SPAN + 2.6, H0 + 50], [-248, RW_SPAN + 2.6, H0 + 102], P), { tone: "faint" }) + D([at([-240, RW_SPAN + 2.6, H0 + 56]), at([-258, RW_SPAN + 2.6, H0 + 56]), at([-240, RW_SPAN + 2.6, H0 + 96])], { size: 0.5 }),
  });
  put({
    id: "rear-structure",
    group: "body",
    xform: "car",
    cls: "fa-floor",
    svg: (() => {
      const out = [];
      const zp = H0 + 4.1;
      out.push(S(k.slabOf({ x: -218, y: -15, w: 62, d: 30, r: 4 }, zp, 38, P, 4, 0.8), { tone: "mid" }));
      out.push(L(k.sideSeam({ x: -218, y: -15, w: 62, d: 30, r: 4 }, zp + 22, P, 4), { tone: "faint" }));
      out.push(D([0.2, 0.5, 0.8].map((f) => at([-218 + 62 * f, 15, zp + 33])), { size: 0.5 }));
      out.push(S(G.disc(-264, -218, 2.4, axisX([0, 0, H0 + 34]), P, { steps: 16 }), { tone: "lo", crease: "none" }));
      out.push(S(k.slabOf({ x: -232, y: -13, w: 14, d: 26, r: 2 }, zp + 36, 12, P, 3, 0.5), { tone: "hi" }));
      return out.join("");
    })(),
  });

  /* --- engine cover, airbox, far sidepod, tub, near sidepod --- */
  const zb = H0 + 4.1;
  const pod = (side) => {
    const out = [];
    const ring = [[30, side * 24], [30, side * 56], [-70, side * 56], [-176, side * 34], [-176, side * 18]];
    out.push(S(k.extrude(ring, zb, 40, P, { bevel: 0.9 }), { tone: "mid" }));
    // inlet on the front face: a dark mouth with a lip
    const m = (y, z) => [30.05, side * y, zb + z];
    out.push(`<path d="${poly3([m(27, 8), m(53, 8), m(53, 30), m(27, 30)])}" style="fill:var(--anatomy-shade-0);stroke:var(--anatomy-lo);stroke-width:var(--anatomy-weight);vector-effect:non-scaling-stroke"/>`);
    out.push(L(poly3([m(29.5, 11), m(50.5, 11), m(50.5, 27), m(29.5, 27)]), { tone: "faint" }));
    out.push(L(k.onTop(ring, zb + 40, P, true), { tone: "faint" }));
    out.push(L(k.ringSeam(ring, zb + 14, P), { tone: "faint" }));
    return out.join("");
  };
  // bargeboards: turning vanes ahead of the sidepod
  const vanes = (side) => [0, 1, 2].map((i) => S(k.slabOf({ x: 42 + i * 5, y: side * (60 + i * 7) - 0.8, w: 26 - i * 4, d: 1.6, r: 0.5 }, zb, 30 - i * 5, P, 2, 0.3), { tone: "mid", crease: "faint" })).join("");
  put({ id: "pod-far", group: "body", xform: "car", cls: "fa-floor", svg: pod(-1) + vanes(-1) });
  put({
    id: "engine",
    group: "body",
    xform: "car",
    cls: "fa-floor",
    svg: (() => {
      const out = [];
      const ring = [[-40, -17], [-40, 17], [-130, 15], [-218, 8], [-218, -8], [-130, -15]];
      out.push(S(k.extrude(ring, zb, 52, P, { bevel: 0.8 }), { tone: "mid" }));
      out.push(L(k.onTop(ring, zb + 52, P, true), { tone: "faint" }));
      // airbox: a scoop above the driver's head
      const F = G.frameOf([0, -13, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
      const scoop = [[-26, zb + 44], [-26, zb + 92], [-48, zb + 98], [-90, zb + 80], [-100, zb + 52]];
      out.push(S(G.prismOf(F, scoop, 0, 26, P), { tone: "hi", crease: "lo" }));
      const mouth = [[-26.05, -9, zb + 66], [-26.05, 9, zb + 66], [-26.05, 9, zb + 88], [-26.05, -9, zb + 88]];
      out.push(`<path d="${poly3(mouth)}" style="fill:var(--anatomy-shade-0);stroke:var(--anatomy-lo);stroke-width:var(--anatomy-weight);vector-effect:non-scaling-stroke"/>`);
      // shark fin
      out.push(S(k.slabOf({ x: -200, y: -1, w: 100, d: 2, r: 0.6 }, zb + 52, 24, P, 2, 0.2), { tone: "mid", crease: "faint" }));
      out.push(D([0.15, 0.4, 0.65, 0.85].map((f) => at([-40 - 178 * f, 0, zb + 52])), { size: 0.5 }));
      return out.join("");
    })(),
  });
  put({
    id: "tub",
    group: "body",
    xform: "car",
    cls: "fa-floor",
    svg: (() => {
      const out = [];
      // monocoque, cockpit opening, nose cone, front wing pylons
      const tub = { x: -36, y: -19, w: 188, d: 38, r: 7 };
      out.push(S(k.slabOf(tub, zb, 46, P, 6, 1), { tone: "hi", crease: "lo" }));
      out.push(L(k.sideSeam(tub, zb + 18, P, 6), { tone: "faint" }));
      const cockpit = { x: 4, y: -13, w: 74, d: 26, r: 8 };
      out.push(`<path d="${k.planOutline(cockpit, zb + 46, P, 6)}" style="fill:var(--anatomy-shade-0);stroke:var(--anatomy-lo);stroke-width:var(--anatomy-weight);vector-effect:non-scaling-stroke"/>`);
      out.push(L(k.planOutline(k.insetPlan(cockpit, 2.4), zb + 46, P, 6), { tone: "faint" }));
      out.push(D(k.corners(tub, 5).map(([x, y]) => at([x, y, zb + 46])), { size: 0.5 }));
      // seat and helmet
      out.push(S(k.slabOf({ x: 4, y: -9, w: 18, d: 18, r: 3 }, zb + 30, 15, P, 3, 0.4), { tone: "mid" }));
      out.push(S(G.sphereOf([40, 0, zb + 59], 10.5, P, { steps: 36 }), { tone: "lit", lit: true, crease: "none" }));
      out.push(L(G.circleOf(G.frameOf([48, 0, zb + 60], [1, 0, 0], [0, 1, 0], [0, 0, 1]), 0, 6.2, P), { tone: "lo" }));
      // steering column and wheel
      out.push(S(k.slabOf({ x: 56, y: -8, w: 2.6, d: 16, r: 1 }, zb + 47, 6, P, 2), { tone: "mid" }));
      // nose cone: a taper along x
      const noseF = axisX([150, 0, zb + 18]);
      out.push(S(G.lathe([[0, 0], [0, 15], [30, 11.5, 1], [70, 8.2, 1], [110, 5.4, 1], [108, 0]], noseF, P, { smooth: true }), { tone: "hi", crease: "lo" }));
      out.push(L(G.arcOf(noseF, 38, 10.6, P, { slope: -0.1 }), { tone: "faint" }));
      // suspension pick-ups
      out.push(D([at([150, 19, zb + 14]), at([150, -19, zb + 14]), at([150, 19, zb + 30]), at([150, -19, zb + 30])], { size: 0.55, tone: "hi" }));
      return out.join("");
    })(),
  });
  put({
    id: "halo",
    group: "body",
    xform: "car",
    cls: "fa-floor",
    svg: (() => {
      const out = [];
      const b = (a, c, r = 1.7) => S(bar(a, c, r, P, { steps: 14 }), { tone: "hi", crease: "none" });
      const top = zb + 80;
      out.push(b([88, 0, zb + 46], [80, 0, top - 2], 2.2));
      [-1, 1].forEach((side) => {
        out.push(b([80, 0, top - 2], [38, side * 21, top], 1.9));
        out.push(b([38, side * 21, top], [8, side * 17, zb + 46], 1.9));
      });
      out.push(b([80, 0, top - 2], [80, 0, top - 2.1], 2.8));
      return out.join("");
    })(),
  });
  put({ id: "pod-near", group: "body", xform: "car", cls: "fa-floor", svg: pod(1) + vanes(1) });

  /* --- near suspension and wheels --- */
  put({ id: "arms-rr", group: "wheels", svg: rearArms(1) });
  put({ id: "arms-fr", group: "wheels", svg: frontArms(1) });
  put({ id: "wheel-rr", group: "wheels", cls: "fa-rubber", svg: wheelStatic(WHEEL_R, 1) });
  put({ id: "wheel-rr-face", group: "wheels", cls: "fa-rubber", deps: ["spin"], live: (pose) => wheelFace(WHEEL_R, 1, pose.spin) });
  put({ id: "wheel-fr", group: "wheels", cls: "fa-rubber", svg: wheelStatic(WHEEL_F, 1) });
  put({ id: "wheel-fr-face", group: "wheels", cls: "fa-rubber", deps: ["spin"], live: (pose) => wheelFace(WHEEL_F, 1, pose.spin) });
  put({
    id: "mirrors",
    group: "body",
    xform: "car",
    cls: "fa-floor",
    svg: [-1, 1].map((side) => S(k.slabOf({ x: 92, y: side * 30, w: 9, d: 7, r: 1.4 }, zb + 52, 3.4, P, 3, 0.4), { tone: "hi" }) + S(k.slabOf({ x: 96, y: side * 22 - 0.7, w: 1.4, d: 8.4, r: 0.4 }, zb + 46, 6, P, 2), { tone: "mid", crease: "none" })).join(""),
  });

  /* --- front wing, then the nose over it --- */
  const FW_X = 242;
  put({
    id: "fw-end-far",
    group: "frontwing",
    xform: "car",
    svg: S(k.slabOf({ x: 200, y: -98 - 2.4, w: 62, d: 2.4, r: 1 }, H0 + 0.6, 24, P, 3, 0.4), { tone: "mid" }) + L(k.segment([230, -100.4, H0 + 1.5], [230, -100.4, H0 + 24.6], P), { tone: "faint" }),
  });
  put({
    id: "fw-planes",
    group: "frontwing",
    xform: "car",
    deps: ["fw"],
    live: (pose) => {
      const out = [];
      const h = H0;
      out.push(wing(foil(FW_X - 3, h + 7.4, 26, 5.5, 6, 0.55), -97, 97, P, { tone: "hi", crease: "lo" }));
      [[222, 12.6, 0.0], [210, 17, 0.0], [197, 21.4, 0.0]].forEach(([x, z, dz], i) => {
        out.push(wing(foil(x, h + z + dz, 19 - i * 1.5, 3.4, pose.fw + 7 - i * 3 + (i ? i * 2 : 0), 0.4), -92, 92, P, { tone: i === 2 ? "lit" : "mid", lit: i === 2 }));
      });
      return out.join("");
    },
  });
  put({
    id: "fw-end-near",
    group: "frontwing",
    xform: "car",
    svg: S(k.slabOf({ x: 200, y: 98, w: 62, d: 2.4, r: 1 }, H0 + 0.6, 24, P, 3, 0.4), { tone: "mid" }) + L(k.segment([230, 100.4, H0 + 1.5], [230, 100.4, H0 + 24.6], P), { tone: "faint" }) + D([at([236, 100.4, H0 + 5]), at([212, 100.4, H0 + 5]), at([236, 100.4, H0 + 19])], { size: 0.5 }),
  });
  put({
    id: "fw-pylons",
    group: "frontwing",
    xform: "car",
    svg: [-1, 1].map((side) => S(k.slabOf({ x: 222, y: side * 11 - 1.3, w: 22, d: 2.6, r: 0.8 }, H0 + 8.5, 14, P, 2, 0.3), { tone: "mid" })).join(""),
  });

  /* --- flow lines above the car, the rake, downforce arrows --- */
  put({
    id: "lines-near",
    group: "flow",
    deps: ["key"],
    live: (pose) =>
      lineTo(upperLine(-10, 24, pose, "wake"), "mid") +
      lineTo(slotLine(0, pose), "lit") +
      lineTo(upperLine(22, 30, pose, ""), "mid") +
      lineTo(upperLine(48, 18, pose, ""), "mid") +
      lineTo(upperLine(122, 12, pose, ""), "mid"),
  });
  put({
    id: "rake",
    group: "rig",
    svg: (() => {
      const out = [];
      const x = 326;
      [-100, 100].forEach((y) => out.push(S(k.cylinder(x, y, 3.4, 0, 30, P, 16), { tone: "mid", crease: "none" })));
      out.push(S(G.disc(-104, 104, 2.4, axisY([x, 0, 30]), P, { steps: 14 }), { tone: "hi", crease: "none" }));
      [-118, -74, -40, -10, 22, 48, 122].forEach((y) => out.push(S(G.disc(0, 8, 1.5, axisX([x - 8, y, 30]), P, { steps: 10 }), { tone: "lo", crease: "none" })));
      return out.join("");
    })(),
  });
  put({
    id: "arrows",
    group: "forces",
    deps: ["fkey", "carShift"],
    live: (pose) => {
      const f = pose.f;
      if (!f || !f.down) return "";
      const wrapW = (m) => `<g style="--anatomy-weight:1.5px">${m}</g>`;
      const len = (n) => Math.min(95, Math.max(0, n / 1000) * 5.6);
      const arrow = (x, y, zTop, zBottom, tone) => {
        const head = 4.2;
        return (
          L(k.segment([x, y, zTop], [x, y, zBottom], P), { tone, free: "start" }) +
          L(k.segment([x, y, zBottom], [x + head, y, zBottom + head * 1.4], P) + k.segment([x, y, zBottom], [x - head, y, zBottom + head * 1.4], P) + k.segment([x, y, zBottom], [x, y + head, zBottom + head * 1.4], P), { tone, free: true })
        );
      };
      const c = pose.carShift;
      const out = [];
      const pairs = [
        [FW_X - 4, 0, H0 + 14 + c, "frontWing"],
        [-243, 0, H0 + 100 + c, "rearWing"],
        [-75, 69, H0 + 15.2 + c, "floor"],
        [-132, 75, H0 + 7.4 + c, "diffuser"],
      ];
      pairs.forEach(([x, y, zb2, key]) => {
        const l = len(f[key]);
        if (l > 1) out.push(arrow(x, y, zb2 + l, zb2, "lit"));
      });
      // drag: backwards from the rear of the car
      const dl = Math.min(110, (f.drag / 1000) * 14);
      if (dl > 1) {
        const x0 = -270;
        const z0 = H0 + 36 + c;
        out.push(L(k.segment([x0, 0, z0], [x0 - dl, 0, z0], P), { tone: "red", free: "start" }));
        out.push(L(k.segment([x0 - dl, 0, z0], [x0 - dl + 5, 4, z0 + 4], P) + k.segment([x0 - dl, 0, z0], [x0 - dl + 5, -4, z0 + 4], P) + k.segment([x0 - dl, 0, z0], [x0 - dl + 5, 0, z0 - 4], P), { tone: "red", free: true }));
      }
      return wrapW(out.join(""));
    },
  });

  /* --- ride-height gauge: a dial indicator on a stand, its plunger resting on the floor's edge fence --- */
  const DIAL = [10, 138];
  put({
    id: "gauge",
    group: "gauge",
    deps: ["dial", "carShift"],
    live: (pose) => {
      const out = [];
      const [x, y] = DIAL;
      out.push(S(k.slabOf({ x: x - 22, y: y - 12, w: 44, d: 24, r: 4 }, -1.6, 9, P, 4, 0.7), { tone: "mid" }));
      out.push(S(k.cylinder(x, y, 4.2, 7.4, 62, P, 20), { tone: "mid", crease: "none" }));
      out.push(L(k.sideArc(x, y, 4.2, 28, P, 12) + k.sideArc(x, y, 4.2, 48, P, 12), { tone: "faint" }));
      // arm out over the car's side, plunger down onto the fence
      const px = x;
      const py = 67.4;
      out.push(S(k.slabOf({ x: x - 3, y: py - 6, w: 6, d: y - py + 9, r: 1.6 }, 62, 5, P, 3, 0.4), { tone: "mid" }));
      const fenceTop = H0 + 4.1 + 7.5 + pose.carShift;
      out.push(S(k.cylinder(px, py, 5.2, 66, 12, P, 24, 0.5), { tone: "hi", crease: "lo" }));
      const top = 66;
      out.push(S(k.cylinder(px, py, 1.1, fenceTop, top - fenceTop, P, 10), { tone: "lit", crease: "none", lit: true }));
      const R = 5;
      out.push(L(k.ring(px, py, R * 0.62, top + 12, P, 24), { tone: "lo" }));
      const ang = (pose.dial / 100) * Math.PI * 1.7 - Math.PI * 0.85;
      out.push(L(k.lineOnTop([px - Math.sin(ang) * 1.2, py + Math.cos(ang) * 1.2], [px + Math.sin(ang) * R * 0.6, py - Math.cos(ang) * R * 0.6], top + 12.2, P), { tone: "lit", free: "start" }));
      return out.join("");
    },
  });
  return list;
}

export const markup = (item, pose) => (item.live ? item.live(pose) : item.svg || "");

export function shiftOf(item, pose) {
  return item.xform === "car" ? [0, 0, pose.carShift] : [0, 0, 0];
}

/** The two dynamic wheel centres and the rear-wing hit area, in screen units, for pointer work. */
export function hitAreas(P) {
  return {
    rearWing: k.iso([-245, 0, H0 + 80], P),
    car: k.iso([0, 0, 40], P),
  };
}

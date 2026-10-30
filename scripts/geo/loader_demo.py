"""The "track being drawn" loading animation, standalone (a prototype; not in the app yet).

    python scripts/geo/loader_demo.py silverstone               # live window, Esc quits
    python scripts/geo/loader_demo.py silverstone --source osm  # from the OSM ways only
    python scripts/geo/loader_demo.py silverstone --record OUT  # frames, GIF and WebP in OUT

Draws from a baked ``f1sim.geo`` asset (``scripts/geo/bake.py``) with ``arcade.gl``, like the
menu: a faint wireframe of the terrain around the circuit (heights exaggerated
``EXAGGERATION`` times), the OSM raceway ways as a ghost, and the circuit's ribbon drawn from
the start line with a bright red head, under a camera ``ELEVATION_DEG`` above the horizon
that orbits once every ``ORBIT_LOOPS`` loops. One loop is ``LOOP_S`` seconds: the ribbon
draws, holds, and fades.

``--source osm`` draws the circuit's loop as found in the OSM ways alone (``loop_xy``, heights
from the terrain grid) instead of the built centre line: the fallback for a circuit that
hasn't been built yet (the menu's "Get track data").

Where it would plug in: the menu's background jobs (Get real lap / Get track data, D44,
:mod:`f1sim.app.jobs`) while they run, and the training screen before the first
``generation.npz`` arrives (:mod:`f1sim.app.training`).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ASSET_DIR = ROOT / "src" / "f1sim" / "app" / "assets" / "geo"
LOOP_S = 3.6
DRAW_S = 2.5  # of LOOP_S: drawing; then hold, then fade
FADE_S = 0.5
ORBIT_LOOPS = 4
ELEVATION_DEG = 35.0
FOV_DEG = 28.0
EXAGGERATION = 4.0
RIBBON_M = 26.0  # wider than the real 12-15 m so it reads at a placeholder's size
TRAIL_M = 260.0  # the head's red fades into the ribbon over this distance

BG = (0x13 / 255, 0x13 / 255, 0x13 / 255)
ACCENT = (0xEB / 255, 0x0D / 255, 0x3F / 255)

_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in float in_s;
out float v_s;
out vec2 v_xy;
void main() { v_s = in_s; v_xy = in_pos.xy; gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""

_GRID_FS = """
#version 330
uniform vec2 u_centre;
uniform float u_radius;
uniform float u_alpha;
in float v_s;
in vec2 v_xy;
out vec4 f;
void main() {
    float r = length(v_xy - u_centre) / u_radius;
    float a = u_alpha * (1.0 - smoothstep(0.55, 1.0, r));
    f = vec4(vec3(1.0) * a, a);   // premultiplied
}
"""

_RIBBON_FS = """
#version 330
uniform float u_head;      // drawn up to this arc length, metres
uniform float u_trail;
uniform float u_fade;      // 1 = visible
uniform vec3 u_accent;
in float v_s;
in vec2 v_xy;
out vec4 f;
void main() {
    if (v_s > u_head) discard;
    float k = exp(-(u_head - v_s) / u_trail);
    vec3 c = mix(vec3(0.80), u_accent, k) + vec3(1.0, 0.85, 0.85) * pow(k, 24.0) * 0.8;
    float a = u_fade * (0.75 + 0.25 * k);
    f = vec4(c * a, a);
}
"""

_GLOW_VS = """
#version 330
uniform vec4 u_clip;       // the head in clip space
uniform vec2 u_half;       // the sprite's half size in NDC
in vec2 in_corner;
out vec2 v_uv;
void main() {
    v_uv = in_corner;
    vec2 ndc = u_clip.xy / u_clip.w + in_corner * u_half;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

_GLOW_FS = """
#version 330
uniform vec3 u_accent;
uniform float u_alpha;
in vec2 v_uv;
out vec4 f;
void main() {
    float d = min(length(v_uv), 1.0);
    float core = exp(-d * d * 40.0);
    float halo = exp(-d * d * 5.0) * (1.0 - d);
    vec3 c = u_accent * halo * 0.9 + vec3(1.0) * core;
    float a = clamp(max(halo * 0.6, core), 0.0, 1.0) * u_alpha;
    f = vec4(c * u_alpha, a);
}
"""


def load_asset(track_id: str) -> dict:
    with np.load(ASSET_DIR / f"{track_id}.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if str(a["format"]) != "f1sim.geo" or int(a["version"]) != 1:
        raise SystemExit("not an f1sim.geo v1 asset")
    return a


class Heights:
    """Bilinear heights (metres above the asset's base) from the coarse grid."""

    def __init__(self, a: dict):
        self.z = a["grid_rel_m"].astype(np.float64)
        self.o, self.step = a["grid_origin"], float(a["grid_step_m"])

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        ny, nx = self.z.shape
        c = np.clip((xy[:, 0] - self.o[0]) / self.step, 0, nx - 1.001)
        r = np.clip((xy[:, 1] - self.o[1]) / self.step, 0, ny - 1.001)
        c0, r0 = c.astype(int), r.astype(int)
        fc, fr = c - c0, r - r0
        z = self.z
        return (z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc) * (1 - fr) + (
            z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc
        ) * fr


def ribbon_strip(xy: np.ndarray, z: np.ndarray, s: np.ndarray, width: float, closed: bool):
    """Triangle-strip vertices (x, y, z, s) for a ribbon along a polyline."""
    if closed:
        xy, z, s = (
            np.vstack([xy, xy[:1]]),
            np.r_[z, z[0]],
            np.r_[s, s[-1] + np.hypot(*(xy[0] - xy[-1]))],
        )
    d = np.gradient(xy, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    left, right = xy + n * width / 2, xy - n * width / 2
    v = np.empty((2 * len(xy), 4), dtype=np.float32)
    v[0::2, :2], v[1::2, :2] = left, right
    v[0::2, 2] = v[1::2, 2] = z
    v[0::2, 3] = v[1::2, 3] = s
    return v


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_deg, aspect, near, far) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


class LoaderScene:
    def __init__(self, ctx, asset: dict, source: str = "track"):
        from arcade.gl import BufferDescription

        self.ctx = ctx
        heights = Heights(asset)
        lift = 1.5  # metres (before exaggeration) above the terrain, so lines don't z-fight

        centre = asset["centre_xy"].astype(np.float64)
        lo, hi = centre.min(0), centre.max(0)
        self.centre = (lo + hi) / 2
        self.radius = float(np.linalg.norm(hi - lo) / 2)
        zc = float(np.median(heights(centre))) * EXAGGERATION
        self.target = np.array([*self.centre, zc])

        def geometry(prog_attrs, data, fmt, names, mode):
            buf = ctx.buffer(data=np.ascontiguousarray(data, dtype=np.float32).tobytes())
            return ctx.geometry([BufferDescription(buf, fmt, names)], mode=mode)

        # terrain wireframe: grid lines in both directions, faded by distance from the centre
        z = asset["grid_rel_m"].astype(np.float64) * EXAGGERATION
        ny, nx = z.shape
        o, st = asset["grid_origin"], float(asset["grid_step_m"])
        gx, gy = o[0] + st * np.arange(nx), o[1] + st * np.arange(ny)
        segs = []
        for j in range(ny):
            p = np.stack([gx, np.full(nx, gy[j]), z[j]], 1)
            segs.append(np.stack([p[:-1], p[1:]], 1).reshape(-1, 3))
        for i in range(nx):
            p = np.stack([np.full(ny, gx[i]), gy, z[:, i]], 1)
            segs.append(np.stack([p[:-1], p[1:]], 1).reshape(-1, 3))
        g = np.vstack(segs)
        g = np.hstack([g, np.zeros((len(g), 1))])
        self.grid = geometry(None, g, "3f 1f", ["in_pos", "in_s"], ctx.LINES)

        # OSM ways as a ghost: line segments; and per-way strips for --source osm
        osm = asset["osm_xy"].astype(np.float64)
        breaks = np.flatnonzero(np.isnan(osm[:, 0]))
        ways = [w[~np.isnan(w[:, 0])] for w in np.split(osm, breaks)]
        ways = [w for w in ways if len(w) >= 2]
        ghost = []
        for w in ways:
            p = np.hstack([w, ((heights(w) + lift) * EXAGGERATION)[:, None]])
            ghost.append(np.stack([p[:-1], p[1:]], 1).reshape(-1, 3))
        g = np.vstack(ghost)
        g = np.hstack([g, np.zeros((len(g), 1))])
        self.ghost = geometry(None, g, "3f 1f", ["in_pos", "in_s"], ctx.LINES)

        # the ribbon(s): s in metres along the line (track) or a fraction (osm, scaled to
        # the circuit's length so the head moves at the same pace)
        self.length = float(asset["length_m"])
        if source == "track":
            line = centre
            elev = asset["elev_rel_m"].astype(np.float64)
            s = np.arange(len(line)) * (self.length / len(line))
            zz = (np.interp(s, np.arange(len(elev)), elev) + lift) * EXAGGERATION
        else:  # the loop found in the OSM ways alone; heights from the grid
            line = asset["loop_xy"].astype(np.float64)
            d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
            s = d / (d[-1] + np.hypot(*(line[0] - line[-1]))) * self.length
            zz = (heights(line) + lift) * EXAGGERATION
        self.head_path = (line, zz, s)
        strips = [ribbon_strip(line, zz, s, RIBBON_M, closed=True)]
        self.ribbons = [
            geometry(None, v, "3f 1f", ["in_pos", "in_s"], ctx.TRIANGLE_STRIP) for v in strips
        ]
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.head = ctx.geometry(
            [BufferDescription(ctx.buffer(data=quad.tobytes()), "2f", ["in_corner"])],
            mode=ctx.TRIANGLE_STRIP,
        )
        self.p_grid = ctx.program(vertex_shader=_VS, fragment_shader=_GRID_FS)
        self.p_ribbon = ctx.program(vertex_shader=_VS, fragment_shader=_RIBBON_FS)
        self.p_glow = ctx.program(vertex_shader=_GLOW_VS, fragment_shader=_GLOW_FS)

    def render(self, t: float, size: tuple[int, int]) -> None:
        ctx = self.ctx
        w, h = size
        loop_t = t % LOOP_S
        az = math.radians(-60.0) + 2 * math.pi * t / (LOOP_S * ORBIT_LOOPS)
        el = math.radians(ELEVATION_DEG)
        dist = self.radius / math.tan(math.radians(FOV_DEG) / 2) * 0.82
        eye = self.target + dist * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        mvp = perspective(FOV_DEG, w / h, dist * 0.2, dist * 3) @ look_at(eye, self.target)
        mvp_t = tuple(mvp.T.ravel().astype(np.float32).tolist())

        draw = _ease(loop_t / DRAW_S)
        head = draw * self.length
        fade = 1.0 - _ease((loop_t - (LOOP_S - FADE_S)) / FADE_S)

        ctx.enable(ctx.BLEND)
        ctx.blend_func = (ctx.ONE, ctx.ONE_MINUS_SRC_ALPHA)
        ctx.disable(ctx.DEPTH_TEST)
        for prog in (self.p_grid, self.p_ribbon):
            prog["u_mvp"] = mvp_t
        self.p_grid["u_centre"] = tuple(self.centre.tolist())
        self.p_grid["u_radius"] = self.radius * 1.35
        self.p_grid["u_alpha"] = 0.10
        self.grid.render(self.p_grid)
        self.p_grid["u_alpha"] = 0.16
        self.ghost.render(self.p_grid)

        self.p_ribbon["u_head"] = head
        self.p_ribbon["u_trail"] = TRAIL_M
        self.p_ribbon["u_fade"] = fade
        self.p_ribbon["u_accent"] = ACCENT
        for r in self.ribbons:
            r.render(self.p_ribbon)

        if fade > 0.01:
            xy, zz, s = self.head_path
            hx = np.interp(head, s, xy[:, 0])
            hy = np.interp(head, s, xy[:, 1])
            hz = np.interp(head, s, zz)
            clip = mvp @ np.array([hx, hy, hz, 1.0])
            px = max(h * 0.045, 3.0)  # half size, pixels
            self.p_glow["u_clip"] = tuple(clip.tolist())
            self.p_glow["u_half"] = (2 * px / w, 2 * px / h)
            self.p_glow["u_accent"] = ACCENT
            # the head glows while drawing, then dims as the lap closes
            self.p_glow["u_alpha"] = fade * (1.0 - 0.7 * _ease((loop_t - DRAW_S) / 0.3))
            self.head.render(self.p_glow)
        ctx.blend_func = ctx.BLEND_DEFAULT


def record(scene_factory, out: Path, size=(480, 300), fps=25, ss=2) -> None:
    """One full orbit (``ORBIT_LOOPS`` loops), rendered at ``ss`` times the size and
    downsampled, saved as WebP (the full orbit) and GIF (one loop) plus a few PNG stills."""
    import arcade
    from PIL import Image

    win = arcade.Window(size[0], size[1], "loader", visible=False)
    ctx = win.ctx
    scene = scene_factory(ctx)
    big = (size[0] * ss, size[1] * ss)
    tex = ctx.texture(big, components=4)
    fbo = ctx.framebuffer(color_attachments=[tex])
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    n = round(LOOP_S * ORBIT_LOOPS * fps)
    stills = {round(fps * t) for t in (0.6, 1.3, 2.1, 2.9)}
    for i in range(n):
        fbo.use()
        fbo.clear(color=(*BG, 1.0))
        scene.render(i / fps, big)
        img = Image.frombytes("RGBA", big, fbo.read(components=4)).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )
        img = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        frames.append(img)
        if i in stills:
            img.save(out / f"frame_{i / fps:04.1f}s.png")
    ms = round(1000 / fps)
    frames[0].save(
        out / "loader.webp",
        save_all=True,
        append_images=frames[1:],
        duration=ms,
        loop=0,
        quality=80,
        method=4,
    )
    one = frames[: round(LOOP_S * fps)]  # GIF: one loop only, it's ~4x the WebP's size
    one[0].save(
        out / "loader.gif", save_all=True, append_images=one[1:], duration=ms, loop=0, optimize=True
    )
    win.close()
    for f in sorted(out.iterdir()):
        print(f"{f} ({f.stat().st_size / 1024:.0f} kB)")


def live(scene_factory, size=(480, 300)) -> None:
    import time

    import arcade

    class W(arcade.Window):
        def __init__(self):
            super().__init__(size[0], size[1], "f1sim loader", resizable=True)
            self.scene = scene_factory(self.ctx)
            self.t0 = time.perf_counter()

        def on_draw(self):
            self.clear(color=tuple(int(255 * c) for c in BG))
            self.scene.render(time.perf_counter() - self.t0, self.get_framebuffer_size())

        def on_key_press(self, key, modifiers):
            if key == arcade.key.ESCAPE:
                self.close()

    W()
    arcade.run()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("track_id", nargs="?", default="silverstone")
    ap.add_argument("--source", choices=("track", "osm"), default="track")
    ap.add_argument("--record", type=Path, metavar="DIR")
    args = ap.parse_args(argv)
    asset = load_asset(args.track_id)

    def factory(ctx):
        return LoaderScene(ctx, asset, args.source)

    if args.record:
        record(factory, args.record)
    else:
        live(factory)


if __name__ == "__main__":
    main()

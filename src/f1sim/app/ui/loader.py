"""The "track being drawn" loader (D49): a circuit drawn in minimalist 3D while data loads.

A faint wireframe of the land around the circuit (heights exaggerated ``EXAGGERATION`` times),
the OSM raceway ways as a ghost, and the circuit's ribbon drawn from the start line with a red
head, under a camera ``ELEVATION_DEG`` above the horizon that orbits once every
``ORBIT_LOOPS`` loops. One loop is ``LOOP_S`` seconds: the ribbon draws, holds and fades.

It draws from a small baked asset per circuit (``assets/geo/<track_id>.npz``, made by
``scripts/geo/bake.py`` from OpenStreetMap and AWS Terrain Tiles), with ``arcade.gl`` into
whatever framebuffer and viewport are bound. ``source="osm"`` draws the circuit's loop as
found in the OSM ways alone instead of the built centre line. The menu shows it over the hero
while a Get track data / Get real lap job runs (:mod:`f1sim.app.menu`);
``scripts/geo/loader_demo.py`` runs it standalone and records it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "geo"
LOOP_S = 5.4
DRAW_S = 3.75  # of LOOP_S: drawing; then hold, then fade
FADE_S = 0.75
HOLD_S = 0.3
"""As a loading screen (:class:`f1sim.app.loading.Opening`) the circuit draws once, then holds
still this long (the head's glow settles in 0.3 s) before the next screen is built."""
ORBIT_LOOPS = 4
ELEVATION_DEG = 35.0
FOV_DEG = 28.0
FIT_ASPECT = 1.6
"""The circuit fills a frame this wide (width / height); narrower frames show it smaller."""
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


_BACK_VS = """
#version 330
in vec2 in_corner;
void main() { gl_Position = vec4(in_corner, 0.0, 1.0); }
"""

_BACK_FS = """
#version 330
uniform vec4 u_colour;     // premultiplied
out vec4 f;
void main() { f = u_colour; }
"""


def asset_path(track_id: str) -> Path:
    return ASSET_DIR / f"{track_id}.npz"


def has_asset(track_id: str) -> bool:
    return asset_path(track_id).is_file()


def asset_from_track(track_id: str, x: np.ndarray, y: np.ndarray, length_m: float) -> dict:
    """A loader asset for a circuit without a baked one: its built centre line (``x``, ``y``
    from s = 0, metres) over flat ground, no OSM ghost."""
    xy = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=1)
    closed = np.vstack([xy, xy[:1]])
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    s = np.linspace(0.0, d[-1], max(int(d[-1] // 4.0), 8), endpoint=False)
    centre = np.stack([np.interp(s, d, closed[:, 0]), np.interp(s, d, closed[:, 1])], axis=1)
    margin = 250.0
    lo, hi = centre.min(0) - margin, centre.max(0) + margin
    step = float(np.ceil((hi - lo).max() / 48))
    nx, ny = (np.ceil((hi - lo) / step).astype(int) + 1).tolist()
    return {
        "track_id": np.array(track_id),
        "centre_xy": centre.astype(np.float32),
        "length_m": np.array(float(length_m)),
        "elev_rel_m": np.zeros(int(np.ceil(length_m)), dtype=np.float16),
        "grid_origin": lo,
        "grid_step_m": np.array(step),
        "grid_rel_m": np.zeros((ny, nx), dtype=np.float16),
        "osm_xy": np.zeros((0, 2), dtype=np.float32),
        "loop_xy": centre.astype(np.float32),
    }


def asset_for(track_id: str, track=None) -> dict:
    """The baked asset if there is one, else one from ``track`` (a :class:`f1sim.track.Track`
    or :class:`~f1sim.track.track.TrackData`); ValueError if neither."""
    if has_asset(track_id):
        return load_asset(track_id)
    data = getattr(track, "data", track)
    if data is None:
        raise ValueError(f"{track_id}: no loader asset and no track")
    return asset_from_track(track_id, data.x, data.y, data.length_m)


def opening_clock(t: float) -> tuple[float, float]:
    """``(loader_t, orbit_t)`` for the loader as a loading screen, ``t`` seconds in: the
    circuit draws once and then holds, and the orbit starts at the loop's speed and slows
    evenly to a stop as the lap closes, so the held frame is still."""
    loader_t = min(t, DRAW_S + HOLD_S)
    u = min(t, DRAW_S)
    return loader_t, u - u * u / (2.0 * DRAW_S)


def load_asset(track_id: str) -> dict:
    """The baked ``f1sim.geo`` v1 asset (``docs/data-formats.md`` section 7)."""
    with np.load(asset_path(track_id), allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if str(a["format"]) != "f1sim.geo" or int(a["version"]) != 1:
        raise ValueError(f"{track_id}: not an f1sim.geo v1 asset")
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
        self.ghost = None
        if ghost:
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
        self.p_back = ctx.program(vertex_shader=_BACK_VS, fragment_shader=_BACK_FS)

    def backdrop(self, alpha: float, rgb=BG) -> None:
        """Cover the bound viewport with ``rgb`` at ``alpha`` (its own shader, so it doesn't
        depend on the projection whatever drew before left behind)."""
        ctx = self.ctx
        ctx.enable(ctx.BLEND)
        ctx.blend_func = (ctx.ONE, ctx.ONE_MINUS_SRC_ALPHA)
        ctx.disable(ctx.DEPTH_TEST)
        self.p_back["u_colour"] = (*(c * alpha for c in rgb), alpha)
        self.head.render(self.p_back)
        ctx.blend_func = ctx.BLEND_DEFAULT

    def render(
        self, t: float, size: tuple[int, int], alpha: float = 1.0, orbit_t: float | None = None
    ) -> None:
        """Draw frame ``t`` (seconds) into the bound viewport of ``size`` pixels; ``alpha``
        fades the whole loader in and out. The camera's orbit follows ``orbit_t`` (default
        ``t``; see :func:`opening_clock`)."""
        ctx = self.ctx
        w, h = size
        loop_t = t % LOOP_S
        orbit = t if orbit_t is None else orbit_t
        az = math.radians(-60.0) + 2 * math.pi * orbit / (LOOP_S * ORBIT_LOOPS)
        el = math.radians(ELEVATION_DEG)
        # framed for a FIT_ASPECT frame; narrower ones back off so the circuit's width fits
        fit = 0.82 * max(1.0, FIT_ASPECT / (w / h))
        dist = self.radius / math.tan(math.radians(FOV_DEG) / 2) * fit
        eye = self.target + dist * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        mvp = perspective(FOV_DEG, w / h, dist * 0.2, dist * 3) @ look_at(eye, self.target)
        mvp_t = tuple(mvp.T.ravel().astype(np.float32).tolist())

        draw = _ease(loop_t / DRAW_S)
        head = draw * self.length
        fade = (1.0 - _ease((loop_t - (LOOP_S - FADE_S)) / FADE_S)) * alpha

        ctx.enable(ctx.BLEND)
        ctx.blend_func = (ctx.ONE, ctx.ONE_MINUS_SRC_ALPHA)
        ctx.disable(ctx.DEPTH_TEST)
        for prog in (self.p_grid, self.p_ribbon):
            prog["u_mvp"] = mvp_t
        self.p_grid["u_centre"] = tuple(self.centre.tolist())
        self.p_grid["u_radius"] = self.radius * 1.35
        self.p_grid["u_alpha"] = 0.10 * alpha
        self.grid.render(self.p_grid)
        self.p_grid["u_alpha"] = 0.16 * alpha
        if self.ghost is not None:
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

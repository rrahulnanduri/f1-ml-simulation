"""Frosted-glass rendering: the scene goes to an offscreen texture, is blurred once per frame,
and each HUD panel samples the blur inside a rounded-rect mask (arcade 3.3 / OpenGL 3.3).

Per frame::

    with glass.scene():        # offscreen framebuffer, cleared to the ambient washes
        ...draw track and cars...
    glass.present()            # one separable Gaussian blur at half resolution, then the
                               # sharp scene is copied to the screen
    glass.panel(rect, radius)  # shadow + blurred backdrop + tint + edge, one draw each

The panel shader does everything in one pass from a rounded-rect SDF: the soft drop shadow
(an SDF falloff, offset ``SHADOW_DY_PX`` down), the backdrop (blurred scene x
``BLUR_BRIGHTEN``), the vertical white tint and the 1 px diagonal-gradient edge. Output is
premultiplied alpha.

**Fallback.** If the framebuffers or programs can't be created (or ``F1SIM_NO_GLASS=1``),
one warning is logged and :class:`GlassRenderer` draws the scene straight to the screen and
each panel as a flat ``PANEL_FALLBACK`` rounded rectangle with the same gradient edge, using
arcade's own shape drawing. The ambient washes and car capsules have their own small shaders
with polygon fallbacks.

Nothing here touches OpenGL until a :class:`GlassRenderer` is constructed.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterator

import numpy as np

from f1sim.app.ui import theme
from f1sim.app.ui.layout import Rect, capsule_points, rounded_rect_points

log = logging.getLogger(__name__)

_QUAD_VS = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
uniform vec4 u_rect;    // x, y, w, h in window pixels
uniform vec2 u_screen;  // window size in pixels
out vec2 v_px;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    v_px = u_rect.xy + in_uv * u_rect.zw;
    gl_Position = vec4(v_px / u_screen * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FS_VS = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

_COPY_FS = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() { f_color = vec4(textureLod(u_tex, v_uv, 0.0).rgb, 1.0); }
"""

_AMBIENT_FS = """
#version 330
uniform vec2 u_screen;
uniform vec3 u_bg;
uniform vec4 u_cool;    // rgb, opacity
uniform vec4 u_warm;
uniform vec2 u_cool_c;  // centre, fraction of the window
uniform vec2 u_warm_c;
uniform float u_radius; // fraction of the diagonal
in vec2 v_uv;
out vec4 f_color;
float wash(vec2 px, vec2 c) {
    float r = u_radius * length(u_screen);
    float t = clamp(distance(px, c * u_screen) / r, 0.0, 1.0);
    return (1.0 - t) * (1.0 - t);
}
void main() {
    vec2 px = v_uv * u_screen;
    vec3 col = u_bg;
    col = mix(col, u_cool.rgb, u_cool.a * wash(px, u_cool_c));
    col = mix(col, u_warm.rgb, u_warm.a * wash(px, u_warm_c));
    // Dither by +-0.5/255 so the faint gradients don't band.
    float n = fract(sin(dot(px, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
    f_color = vec4(col + n / 255.0, 1.0);
}
"""

_BLUR_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_step;    // one destination texel along the blur direction, in uv
uniform float u_sigma;  // in destination texels
uniform float u_lod;    // source mip level matching the destination resolution
in vec2 v_uv;
out vec4 f_color;
const int RADIUS = 14;
void main() {
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int i = -RADIUS; i <= RADIUS; ++i) {
        float w = exp(-0.5 * float(i * i) / (u_sigma * u_sigma));
        acc += w * textureLod(u_tex, v_uv + float(i) * u_step, u_lod).rgb;
        wsum += w;
    }
    f_color = vec4(acc / wsum, 1.0);
}
"""

_PANEL_FS = """
#version 330
uniform sampler2D u_blur;
uniform vec2 u_screen;
uniform vec4 u_panel;     // x, y, w, h
uniform float u_radius;
uniform float u_brighten;
uniform vec2 u_tint;      // top, bottom white opacity
uniform vec3 u_edge;      // top-left, middle, bottom-right white opacity
uniform vec3 u_shadow;    // dy (down), blur, opacity
in vec2 v_px;
out vec4 f_color;
float sd_round_box(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
void main() {
    vec2 hb = 0.5 * u_panel.zw;
    vec2 c = u_panel.xy + hb;
    float d = sd_round_box(v_px - c, hb, u_radius);
    float inside = clamp(0.5 - d, 0.0, 1.0);

    float ds = sd_round_box(v_px - (c - vec2(0.0, u_shadow.x)), hb, u_radius);
    float shadow = u_shadow.z * (1.0 - smoothstep(-u_shadow.y, u_shadow.y, ds));

    vec3 col = texture(u_blur, v_px / u_screen).rgb * u_brighten;
    float ty = clamp((v_px.y - u_panel.y) / u_panel.w, 0.0, 1.0);
    col = mix(col, vec3(1.0), mix(u_tint.y, u_tint.x, ty));

    vec2 rel = clamp((v_px - u_panel.xy) / u_panel.zw, 0.0, 1.0);
    float g = 0.5 * (rel.x + 1.0 - rel.y);   // 0 top-left .. 1 bottom-right
    float ea = g < 0.5 ? mix(u_edge.x, u_edge.y, g * 2.0) : mix(u_edge.y, u_edge.z, g * 2.0 - 1.0);
    float edge = clamp(1.0 - abs(d + 0.5), 0.0, 1.0);
    col = mix(col, vec3(1.0), edge * ea);

    float a = inside + shadow * (1.0 - inside);
    f_color = vec4(min(col, vec3(1.0)) * inside, a);
}
"""

_CAPSULE_FS = """
#version 330
uniform vec2 u_centre;
uniform float u_heading;
uniform vec2 u_size;      // length, width in pixels
uniform vec4 u_colour;    // straight alpha
uniform float u_stroke;   // 0 = filled, else the outline width in pixels
in vec2 v_px;
out vec4 f_color;
void main() {
    vec2 p = v_px - u_centre;
    float c = cos(u_heading), s = sin(u_heading);
    p = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);
    float r = 0.5 * min(u_size.x, u_size.y);
    float a = max(0.5 * u_size.x - r, 0.0);
    float d = length(vec2(p.x - clamp(p.x, -a, a), p.y)) - r;
    float cov = u_stroke > 0.0
        ? clamp(0.5 * u_stroke + 0.5 - abs(d + 0.5 * u_stroke), 0.0, 1.0)
        : clamp(0.5 - d, 0.0, 1.0);
    float alpha = u_colour.a * cov;
    f_color = vec4(u_colour.rgb * alpha, alpha);
}
"""


def _rgb(c) -> tuple[float, float, float]:
    return c[0] / 255.0, c[1] / 255.0, c[2] / 255.0


def _rgba(c) -> tuple[float, float, float, float]:
    return c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, (c[3] if len(c) > 3 else 255) / 255.0


class GlassRenderer:
    """Offscreen scene, blur and glass panels for one window. See the module docstring."""

    def __init__(self, window, enabled: bool = True):
        self.window = window
        self.ctx = window.ctx
        self._warned = False
        self._fallback_edges: dict[tuple, object] = {}
        self._fallback_fills: dict[tuple, object] = {}
        self._shaders_ok = self._init_shaders()
        self.enabled = bool(enabled) and self._shaders_ok and not theme.glass_disabled_by_env()
        if enabled and not self.enabled and self._shaders_ok:
            self._warn(f"glass disabled by {theme.NO_GLASS_ENV}; using flat panels")
        self._size: tuple | None = None
        if self.enabled:
            try:
                self._ensure_targets()
            except Exception as exc:  # noqa: BLE001 - any GL failure -> flat panels
                self._warn(f"can't create the glass framebuffers ({exc}); using flat panels")
                self.enabled = False

    # ------------------------------------------------------------------------ setup

    def _warn(self, msg: str) -> None:
        if not self._warned:
            log.warning(msg)
            self._warned = True

    def _init_shaders(self) -> bool:
        from arcade.gl import geometry

        try:
            ctx = self.ctx
            self._quad = geometry.quad_2d(size=(1.0, 1.0), pos=(0.5, 0.5))
            self._fs_quad = geometry.quad_2d_fs()
            self._p_copy = ctx.program(vertex_shader=_FS_VS, fragment_shader=_COPY_FS)
            self._p_ambient = ctx.program(vertex_shader=_FS_VS, fragment_shader=_AMBIENT_FS)
            self._p_blur = ctx.program(vertex_shader=_FS_VS, fragment_shader=_BLUR_FS)
            self._p_panel = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_PANEL_FS)
            self._p_capsule = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_CAPSULE_FS)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"can't compile the glass shaders ({exc}); using flat panels")
            return False
        a = self._p_ambient
        a["u_bg"] = _rgb(theme.BACKGROUND)
        a["u_cool"] = _rgba(theme.AMBIENT_COOL)
        a["u_warm"] = _rgba(theme.AMBIENT_WARM)
        a["u_cool_c"] = theme.AMBIENT_COOL_CENTRE
        a["u_warm_c"] = theme.AMBIENT_WARM_CENTRE
        a["u_radius"] = theme.AMBIENT_RADIUS
        p = self._p_panel
        p["u_brighten"] = theme.BLUR_BRIGHTEN
        p["u_tint"] = (theme.TINT_TOP, theme.TINT_BOTTOM)
        p["u_edge"] = (theme.EDGE_TOP_LEFT, theme.EDGE_MIDDLE, theme.EDGE_BOTTOM_RIGHT)
        p["u_shadow"] = (theme.SHADOW_DY_PX, theme.SHADOW_BLUR_PX, theme.SHADOW_ALPHA)
        return True

    def _ensure_targets(self) -> None:
        """Scene target at the physical framebuffer size (2x the window on a 200 % display);
        blur targets at half the *logical* size, so the blur's sigma and cost are the same
        in window pixels at any pixel ratio."""
        size = tuple(int(v) for v in self.window.get_framebuffer_size())
        size = (max(size[0], 1), max(size[1], 1))
        logical = (max(int(self.window.width), 1), max(int(self.window.height), 1))
        if (size, logical) == self._size:
            return
        ctx = self.ctx
        lin = (ctx.LINEAR, ctx.LINEAR)
        clamp = ctx.CLAMP_TO_EDGE
        half = (max(logical[0] // 2, 1), max(logical[1] // 2, 1))
        self._scene_tex = ctx.texture(
            size,
            components=4,
            filter=(ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR),
            wrap_x=clamp,
            wrap_y=clamp,
        )
        self._scene_fbo = ctx.framebuffer(color_attachments=[self._scene_tex])
        # Mip level whose resolution matches the blur target: box-filtered downsample.
        self._lod = max(0.0, math.log2(size[0] / half[0]))
        self._blur_tex = [
            ctx.texture(half, components=4, filter=lin, wrap_x=clamp, wrap_y=clamp)
            for _ in range(2)
        ]
        self._blur_fbo = [ctx.framebuffer(color_attachments=[t]) for t in self._blur_tex]
        self._size = (size, logical)
        self._phys = size
        from arcade.camera import Camera2D

        self._pixel_camera = Camera2D()  # matches the window: projection in window pixels
        self._half = half

    # ------------------------------------------------------------------------ frame

    def _screen_size(self) -> tuple[float, float]:
        return float(self.window.width), float(self.window.height)

    def _draw_ambient(self) -> None:
        if not self._shaders_ok:
            return  # plain background (already cleared)
        self._p_ambient["u_screen"] = self._screen_size()
        self.ctx.disable(self.ctx.BLEND)
        self._fs_quad.render(self._p_ambient)
        self.ctx.enable(self.ctx.BLEND)

    @contextlib.contextmanager
    def scene(self) -> Iterator[None]:
        """Draw the scene inside this block: into the offscreen texture (glass on) or the
        screen (fallback), over the background and ambient washes."""
        win = self.window
        if self.enabled:
            try:
                self._ensure_targets()
            except Exception as exc:  # noqa: BLE001
                self._warn(f"can't resize the glass framebuffers ({exc}); using flat panels")
                self.enabled = False
        if not self.enabled:
            win.clear(color=theme.BACKGROUND)
            self._draw_ambient()
            yield
            return
        with self._scene_fbo.activate():
            self._scene_fbo.clear(color=theme.BACKGROUND)
            self.use_pixels()
            self._draw_ambient()
            yield
        win.default_camera.use()  # back on the screen: arcade applies the pixel ratio

    def use_camera(self, camera) -> None:
        """Activate an arcade ``Camera2D`` for drawing the scene.

        On a high-DPI display the scene texture has the physical size (e.g. 2x the window).
        arcade sets viewports in logical pixels and only scales them for the screen, so
        inside :meth:`scene` the viewport is reset to the whole physical texture after the
        camera is applied; the camera's projection still spans the logical window, so
        everything lands where it would on screen and the texture composites 1:1."""
        camera.use()
        if self.enabled and self.ctx.active_framebuffer is getattr(self, "_scene_fbo", None):
            self._scene_fbo.viewport = (0, 0, *self._phys)
            self._scene_fbo.scissor = None

    def use_pixels(self) -> None:
        """Screen-space (window pixel) projection for scene drawing: start line, cars,
        labels. arcade's default camera isn't used offscreen because it re-derives its
        projection from the active viewport, which is physical-sized there."""
        if self.enabled:
            self.use_camera(self._pixel_camera)
        else:
            self.window.default_camera.use()

    def present(self) -> None:
        """Blur the scene (two passes at half resolution) and copy it to the screen. A no-op
        in fallback mode (the scene is already on screen)."""
        if not self.enabled:
            return
        ctx = self.ctx
        ctx.disable(ctx.BLEND)
        sigma = theme.BLUR_SIGMA_PX * 0.5  # texels of the half-logical-size targets
        hw, hh = self._half
        blur = self._p_blur
        blur["u_sigma"] = sigma
        lod = self._lod
        self._scene_tex.build_mipmaps(max_level=int(math.ceil(lod)) + 1)
        # Horizontal: scene (at the mip level matching the target) -> A; vertical: A -> B.
        with self._blur_fbo[0].activate():
            self._blur_fbo[0].viewport = (0, 0, hw, hh)
            self._scene_tex.use(0)
            blur["u_step"] = (1.0 / hw, 0.0)
            blur["u_lod"] = lod
            self._fs_quad.render(blur)
        with self._blur_fbo[1].activate():
            self._blur_fbo[1].viewport = (0, 0, hw, hh)
            self._blur_tex[0].use(0)
            blur["u_step"] = (0.0, 1.0 / hh)
            blur["u_lod"] = 0.0
            self._fs_quad.render(blur)
        self.window.default_camera.use()
        self._scene_tex.use(0)
        self._fs_quad.render(self._p_copy)
        ctx.enable(ctx.BLEND)

    # ------------------------------------------------------------------------ panels

    def panel(self, rect: Rect, radius: float = theme.PANEL_RADIUS) -> None:
        """One glass panel (or its flat fallback) in window pixels."""
        if self.enabled:
            ctx = self.ctx
            margin = theme.SHADOW_DY_PX + 2.0 * theme.SHADOW_BLUR_PX
            p = self._p_panel
            p["u_screen"] = self._screen_size()
            p["u_panel"] = (rect.x, rect.y, rect.w, rect.h)
            p["u_radius"] = float(radius)
            p["u_rect"] = (
                rect.x - margin,
                rect.y - margin,
                rect.w + 2 * margin,
                rect.h + 2 * margin,
            )
            self._blur_tex[1].use(0)
            ctx.enable(ctx.BLEND)
            ctx.blend_func = ctx.BLEND_PREMULTIPLIED_ALPHA
            self._quad.render(p)
            ctx.blend_func = ctx.BLEND_DEFAULT
            return
        self._flat_panel(rect, radius)

    def _flat_panel(self, rect: Rect, radius: float) -> None:
        from arcade import shape_list

        key = (rect.x, rect.y, rect.w, rect.h, radius)
        shapes = self._fallback_fills.get(key)
        if shapes is None:
            if len(self._fallback_fills) > 32:
                self._fallback_fills.clear()
            pts = rounded_rect_points(rect, radius)
            cx, cy = rect.centre
            shapes = shape_list.ShapeElementList()
            fan = [(cx, cy)] + [tuple(p) for p in pts] + [tuple(pts[0])]
            tris = []
            for i in range(1, len(fan) - 1):
                tris += [fan[0], fan[i], fan[i + 1]]
            shapes.append(
                shape_list.create_triangles_filled_with_colors(
                    tris, [theme.PANEL_FALLBACK] * len(tris)
                )
            )
            # The edge: per-vertex white with the same diagonal gradient as the shader.
            inner = rounded_rect_points(rect.inset(0.5), max(radius - 0.5, 0.0))
            rel_x = (inner[:, 0] - rect.x) / rect.w
            rel_y = (inner[:, 1] - rect.y) / rect.h
            cols = [
                theme.white(theme.edge_alpha(0.5 * (u + 1.0 - v)))
                for u, v in zip(rel_x, rel_y, strict=True)
            ]
            loop = [tuple(p) for p in inner] + [tuple(inner[0])]
            seg_pts, seg_cols = [], []
            for i in range(len(loop) - 1):
                seg_pts += [loop[i], loop[i + 1]]
                seg_cols += [cols[i], cols[(i + 1) % len(cols)]]
            shapes.append(shape_list.create_lines_with_colors(seg_pts, seg_cols))
            self._fallback_fills[key] = shapes
        shapes.draw()

    # ------------------------------------------------------------------------- cars

    def capsule(
        self,
        cx: float,
        cy: float,
        heading: float,
        length_px: float,
        width_px: float,
        colour,
        stroke_px: float = 0.0,
    ) -> None:
        """An anti-aliased capsule in window pixels: filled (``stroke_px = 0``) or outlined."""
        if self._shaders_ok:
            ctx = self.ctx
            half = 0.5 * max(length_px, width_px) + 2.0
            p = self._p_capsule
            p["u_screen"] = self._screen_size()
            p["u_rect"] = (cx - half, cy - half, 2 * half, 2 * half)
            p["u_centre"] = (cx, cy)
            p["u_heading"] = float(heading)
            p["u_size"] = (float(length_px), float(width_px))
            p["u_colour"] = _rgba(colour)
            p["u_stroke"] = float(stroke_px)
            ctx.enable(ctx.BLEND)
            ctx.blend_func = ctx.BLEND_PREMULTIPLIED_ALPHA
            self._quad.render(p)
            ctx.blend_func = ctx.BLEND_DEFAULT
            return
        import arcade

        pts = [tuple(p) for p in capsule_points(cx, cy, heading, length_px, width_px)]
        if stroke_px > 0.0:
            arcade.draw_polygon_outline(pts, colour, max(1.0, stroke_px))
        else:
            arcade.draw_polygon_filled(pts, colour)


def draw_start_line(
    line_world: np.ndarray, camera, colour=theme.START_LINE, width_px: float = theme.START_LINE_PX
) -> None:
    """The start/finish line (``(2, 2)`` world endpoints, from
    :func:`f1sim.app.render_track.boundary_lines`) as a solid screen-space line of
    ``width_px``, under the window's default camera."""
    import arcade

    sx, sy = camera.world_to_screen(line_world[:, 0], line_world[:, 1])
    arcade.draw_line(float(sx[0]), float(sy[0]), float(sx[1]), float(sy[1]), colour, width_px)

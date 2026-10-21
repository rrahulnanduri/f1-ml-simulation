"""Card rendering over a dark, faintly blurred scene (arcade 3.3 / OpenGL 3.3).

The scene goes to an offscreen texture (4x multisampled, then resolved), is blurred once per
frame, and each HUD card is a solid near-black rounded rectangle with a white hairline whose
fill lets only ``1 - CARD_FILL`` of the blurred scene through, so the track faintly shows.

Per frame::

    with glass.scene():        # offscreen framebuffer, cleared to the background + vignette
        ...draw track and cars...
    glass.present()            # resolve, one separable Gaussian blur at half resolution, then
                               # the sharp scene is copied to the screen
    glass.panel(rect)          # one card: blurred backdrop x (1 - fill) + 1 px hairline

The panel shader works from a rounded-rect SDF in window pixels: the fill is exactly inside the
rectangle, the hairline exactly its outermost logical pixel (two physical pixels on a 2x
display), anti-aliased only round the corners. Output is premultiplied alpha.

**Fallback.** If the framebuffers or programs can't be created (or ``F1SIM_NO_GLASS=1``), one
warning is logged and :class:`GlassRenderer` draws the scene straight to the screen and each
card as a flat ``CARD_FALLBACK`` rounded rectangle with the same hairline (triangles). Discs
and strokes (buttons, page dots) use a small SDF shader with polygon fallbacks.

Nothing here touches OpenGL until a :class:`GlassRenderer` is constructed.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterator

from f1sim.app.ui import theme
from f1sim.app.ui.layout import Rect, capsule_points, rounded_rect_points

log = logging.getLogger(__name__)

MSAA_SAMPLES = 4

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
uniform float u_vignette;  // darkening at the corners
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec2 px = v_uv * u_screen;
    float r = length(px - 0.5 * u_screen) / (0.5 * length(u_screen));  // 0 centre, 1 corner
    vec3 col = u_bg * (1.0 - u_vignette * smoothstep(0.35, 1.0, r));
    // Dither by +-0.5/255 so the faint gradient doesn't band.
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
const int RADIUS = 10;
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
uniform float u_fill;     // opacity of the black fill over the blurred scene
uniform vec4 u_border;    // hairline colour, straight alpha
in vec2 v_px;
out vec4 f_color;
float sd_round_box(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
void main() {
    vec2 hb = 0.5 * u_panel.zw;
    float d = sd_round_box(v_px - (u_panel.xy + hb), hb, u_radius);
    float inside = clamp(0.5 - d, 0.0, 1.0);
    vec3 col = texture(u_blur, v_px / u_screen).rgb * (1.0 - u_fill);
    float edge = clamp(1.0 - abs(d + 0.5), 0.0, 1.0);  // the outermost pixel
    col = mix(col, u_border.rgb, edge * u_border.a);
    f_color = vec4(col * inside, inside);
}
"""

_CAPSULE_FS = """
#version 330
uniform vec2 u_centre;
uniform float u_heading;
uniform vec2 u_size;      // length, width in pixels
uniform vec4 u_colour;    // straight alpha
in vec2 v_px;
out vec4 f_color;
void main() {
    vec2 p = v_px - u_centre;
    float c = cos(u_heading), s = sin(u_heading);
    p = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);
    float r = 0.5 * min(u_size.x, u_size.y);
    float a = max(0.5 * u_size.x - r, 0.0);
    float d = length(vec2(p.x - clamp(p.x, -a, a), p.y)) - r;
    float alpha = u_colour.a * clamp(0.5 - d, 0.0, 1.0);
    f_color = vec4(u_colour.rgb * alpha, alpha);
}
"""


def _rgb(c) -> tuple[float, float, float]:
    return c[0] / 255.0, c[1] / 255.0, c[2] / 255.0


def _rgba(c) -> tuple[float, float, float, float]:
    return c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, (c[3] if len(c) > 3 else 255) / 255.0


class GlassRenderer:
    """Offscreen scene, blur and cards for one window. See the module docstring."""

    def __init__(self, window, enabled: bool = True):
        self.window = window
        self.ctx = window.ctx
        self._warned = False
        self._fallback_shapes: dict[tuple, object] = {}
        self._shaders_ok = self._init_shaders()
        self.enabled = bool(enabled) and self._shaders_ok and not theme.glass_disabled_by_env()
        if enabled and not self.enabled and self._shaders_ok:
            self._warn(f"glass disabled by {theme.NO_GLASS_ENV}; using flat panels")
        self._size: tuple | None = None
        self._msaa_fbo = None
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
        a["u_vignette"] = theme.VIGNETTE
        p = self._p_panel
        p["u_fill"] = theme.CARD_FILL
        p["u_border"] = _rgba(theme.CARD_BORDER)
        return True

    def _ensure_targets(self) -> None:
        """Scene targets at the physical framebuffer size (2x the window on a 200 % display):
        a multisampled one to draw into and a plain one it resolves to; blur targets at half
        the *logical* size, so the blur's sigma and cost are the same in window pixels at any
        pixel ratio."""
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
        self._msaa_fbo = None
        samples = min(MSAA_SAMPLES, int(getattr(ctx.info, "MAX_SAMPLES", 0) or 0))
        if samples > 1:
            try:
                self._msaa_tex = ctx.texture(size, components=4, samples=samples)
                self._msaa_fbo = ctx.framebuffer(color_attachments=[self._msaa_tex])
            except Exception as exc:  # noqa: BLE001 - no MSAA: draw into the plain target
                log.info("No multisampled scene target (%s)", exc)
                self._msaa_fbo = None
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

    @property
    def pixel_ratio(self) -> float:
        return float(self.window.get_pixel_ratio())

    def _screen_size(self) -> tuple[float, float]:
        return float(self.window.width), float(self.window.height)

    def _draw_ambient(self) -> None:
        if not self._shaders_ok:
            return  # plain background (already cleared)
        self._p_ambient["u_screen"] = self._screen_size()
        self.ctx.disable(self.ctx.BLEND)
        self._fs_quad.render(self._p_ambient)
        self.ctx.enable(self.ctx.BLEND)

    def _target(self):
        return self._msaa_fbo if self._msaa_fbo is not None else self._scene_fbo

    @contextlib.contextmanager
    def scene(self) -> Iterator[None]:
        """Draw the scene inside this block: into the offscreen texture (glass on) or the
        screen (fallback), over the background and its vignette."""
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
        target = self._target()
        with target.activate():
            target.clear(color=theme.BACKGROUND)
            self.use_pixels()
            self._draw_ambient()
            yield
        if target is not self._scene_fbo:
            self.ctx.copy_framebuffer(target, self._scene_fbo)  # resolve the samples
            self.ctx.active_framebuffer.use(force=True)  # the blit rebinds behind arcade
        win.default_camera.use()  # back on the screen: arcade applies the pixel ratio

    def use_camera(self, camera) -> None:
        """Activate an arcade ``Camera2D`` for drawing the scene.

        On a high-DPI display the scene texture has the physical size (e.g. 2x the window).
        arcade sets viewports in logical pixels and only scales them for the screen, so
        inside :meth:`scene` the viewport is reset to the whole physical texture after the
        camera is applied; the camera's projection still spans the logical window, so
        everything lands where it would on screen and the texture composites 1:1."""
        camera.use()
        if self.enabled:
            fbo = self.ctx.active_framebuffer
            if fbo is self._scene_fbo or (fbo is not None and fbo is self._msaa_fbo):
                fbo.viewport = (0, 0, *self._phys)
                fbo.scissor = None

    def use_pixels(self) -> None:
        """Screen-space (window pixel) projection for scene drawing: cars, labels, ticks.
        arcade's default camera isn't used offscreen because it re-derives its projection
        from the active viewport, which is physical-sized there."""
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
        hw, hh = self._half
        blur = self._p_blur
        blur["u_sigma"] = theme.BLUR_SIGMA_PX * 0.5  # texels of the half-size targets
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

    # ------------------------------------------------------------------------ cards

    def panel(self, rect: Rect, radius: float = theme.CARD_RADIUS) -> None:
        """One card (or its flat fallback) in window pixels."""
        if self.enabled:
            ctx = self.ctx
            p = self._p_panel
            p["u_screen"] = self._screen_size()
            p["u_panel"] = (rect.x, rect.y, rect.w, rect.h)
            p["u_radius"] = float(radius)
            p["u_rect"] = (rect.x - 1.0, rect.y - 1.0, rect.w + 2.0, rect.h + 2.0)
            self._blur_tex[1].use(0)
            ctx.enable(ctx.BLEND)
            ctx.blend_func = (ctx.ONE, ctx.ONE_MINUS_SRC_ALPHA)  # premultiplied "over"
            self._quad.render(p)
            ctx.blend_func = ctx.BLEND_DEFAULT
            return
        self._flat_panel(rect, radius)

    def _flat_panel(self, rect: Rect, radius: float) -> None:
        from arcade import shape_list

        key = (rect.x, rect.y, rect.w, rect.h, radius)
        shapes = self._fallback_shapes.get(key)
        if shapes is None:
            if len(self._fallback_shapes) > 32:
                self._fallback_shapes.clear()
            outer = rounded_rect_points(rect, radius)
            inner = rounded_rect_points(rect.inset(1.0), max(radius - 1.0, 0.0))
            cx, cy = rect.centre
            fill = []
            for i in range(len(inner)):
                j = (i + 1) % len(inner)
                fill += [(cx, cy), tuple(inner[i]), tuple(inner[j])]
            ring = []
            for i in range(len(outer)):
                j = (i + 1) % len(outer)
                a, b, c, d = tuple(outer[i]), tuple(outer[j]), tuple(inner[i]), tuple(inner[j])
                ring += [a, b, c, b, d, c]
            # The hairline over the fill, blended as the shader does.
            edge = _over(theme.CARD_BORDER, theme.CARD_FALLBACK)
            shapes = shape_list.ShapeElementList()
            shapes.append(
                shape_list.create_triangles_filled_with_colors(
                    fill, [theme.CARD_FALLBACK] * len(fill)
                )
            )
            shapes.append(shape_list.create_triangles_filled_with_colors(ring, [edge] * len(ring)))
            self._fallback_shapes[key] = shapes
        shapes.draw()

    # ------------------------------------------------------------------- discs, strokes

    def capsule(
        self, cx: float, cy: float, heading: float, length_px: float, width_px: float, colour
    ) -> None:
        """An anti-aliased filled capsule in window pixels."""
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
            ctx.enable(ctx.BLEND)
            ctx.blend_func = (ctx.ONE, ctx.ONE_MINUS_SRC_ALPHA)  # premultiplied "over"
            self._quad.render(p)
            ctx.blend_func = ctx.BLEND_DEFAULT
            return
        import arcade

        arcade.draw_polygon_filled(
            [tuple(p) for p in capsule_points(cx, cy, heading, length_px, width_px)], colour
        )

    def disc(self, cx: float, cy: float, r: float, colour) -> None:
        self.capsule(cx, cy, 0.0, 2 * r, 2 * r, colour)

    def stroke(self, x0: float, y0: float, x1: float, y1: float, width: float, colour) -> None:
        """A line with round caps, ``width`` px wide."""
        length = math.hypot(x1 - x0, y1 - y0) + width
        heading = math.atan2(y1 - y0, x1 - x0)
        self.capsule(0.5 * (x0 + x1), 0.5 * (y0 + y1), heading, length, width, colour)


def _over(top, bottom) -> tuple[int, int, int, int]:
    """``top`` composited over ``bottom`` (straight alpha), keeping ``bottom``'s opacity."""
    a = top[3] / 255.0
    return (
        int(round(top[0] * a + bottom[0] * (1 - a))),
        int(round(top[1] * a + bottom[1] * (1 - a))),
        int(round(top[2] * a + bottom[2] * (1 - a))),
        max(bottom[3], top[3]),
    )

"""The "track being drawn" loader (:mod:`f1sim.app.ui.loader`, D49) on its own.

    python scripts/geo/loader_demo.py silverstone               # live window, Esc quits
    python scripts/geo/loader_demo.py silverstone --source osm  # from the OSM loop alone
    python scripts/geo/loader_demo.py silverstone --record OUT  # frames, GIF and WebP in OUT

``--source osm`` draws the circuit's loop as found in the OSM ways alone (``loop_xy``, heights
from the terrain grid) instead of the built centre line: the fallback for a circuit that
hasn't been built yet. In the app the menu shows it over the hero while a Get track data /
Get real lap job runs, and ``L`` on the menu previews it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from f1sim.app.ui.loader import BG, LOOP_S, ORBIT_LOOPS, LoaderScene, load_asset  # noqa: E402


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

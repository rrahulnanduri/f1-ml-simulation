"""Copy a light-stage render into the app's bundled assets, compressed.

    python scripts/hero/export_stage.py [stage_dir]
    python scripts/hero/export_stage.py --drive [drive_stage_dir]

Reads ``render_top.py --stage`` output (default ``assets_src/sf26/render/stage``: the
``light_<row>_<col>.png`` photos, ``ambient.png`` and ``stage.json``) and writes them to
``src/f1sim/app/assets/hero/stage`` as WebP (quality ``QUALITY``; the ambient photo keeps its
alpha). That's what the menu shows (:mod:`f1sim.app.ui.light_stage`), on every checkout and in
the installed package: the 45 photos of the black car come to under 1 MB.

``--drive`` exports ``render_top.py --drive`` output (default
``assets_src/sf26/render/drive_stage``) to ``src/f1sim/app/assets/hero/drive`` for the driving
hero: ``body_*`` and ``body_ambient`` as RGBA WebP (the ambient photo's alpha carries the
soft ground shadow), and the front-wheel photos cropped to the strip of rows around the
wheels (``WHEEL_PAD_M`` beyond their radius), recorded as ``wheel_rows`` in ``stage.json``.

Re-run it after re-rendering, then commit the folder.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets_src" / "sf26" / "render" / "stage"
DEST = ROOT / "src" / "f1sim" / "app" / "assets" / "hero" / "stage"
DRIVE_SOURCE = ROOT / "assets_src" / "sf26" / "render" / "drive_stage"
DRIVE_DEST = ROOT / "src" / "f1sim" / "app" / "assets" / "hero" / "drive"
QUALITY = 95
WHEEL_PAD_M = 0.25


def main(source: Path = SOURCE) -> int:
    info_path = source / "stage.json"
    if not info_path.is_file():
        print(f"no stage.json in {source}: render it first (render_top.py --stage)")
        return 1
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not info.get("complete"):
        print(f"{source} is a test render (--stage-test): render the full stage first")
        return 1
    names = [f"light_{r}_{c}" for r in range(len(info["ys"])) for c in range(len(info["xs"]))]
    missing = [n for n in [*names, "ambient"] if not (source / f"{n}.png").is_file()]
    if missing:
        print(f"missing in {source}: {', '.join(missing)}")
        return 1
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    total = 0
    for name in names:
        out = DEST / f"{name}.webp"
        Image.open(source / f"{name}.png").convert("RGB").save(
            out, "WEBP", quality=QUALITY, method=6
        )
        total += out.stat().st_size
    out = DEST / "ambient.webp"
    Image.open(source / "ambient.png").convert("RGBA").save(
        out, "WEBP", quality=QUALITY, method=6, exact=True
    )
    total += out.stat().st_size
    (DEST / "stage.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"wrote {len(names) + 1} photos to {DEST.relative_to(ROOT)} ({total / 1e6:.2f} MB)")
    return 0


def export_drive(source: Path = DRIVE_SOURCE) -> int:
    info_path = source / "stage.json"
    if not info_path.is_file():
        print(f"no stage.json in {source}: render it first (render_top.py --drive)")
        return 1
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not info.get("complete"):
        print(f"{source} is a test render (--stage-test): render the full stage first")
        return 1
    cells = [f"{r}_{c}" for r in range(len(info["ys"])) for c in range(len(info["xs"]))]
    names = [f"{layer}_{c}" for layer in ("body", "wheel") for c in cells]
    names += ["body_ambient", "wheel_ambient"]
    missing = [n for n in names if not (source / f"{n}.png").is_file()]
    if missing:
        print(f"missing in {source}: {', '.join(missing[:5])}...")
        return 1
    ppm = float(info["px_per_m"])
    ys = [w["y"] for w in info["wheels"]]
    r = max(w["r"] for w in info["wheels"]) + WHEEL_PAD_M
    row0 = max(int((min(ys) - r) * ppm), 0)
    row1 = min(int((max(ys) + r) * ppm) + 1, int(info["image_px"][1]))
    info["wheel_rows"] = [row0, row1 - row0]
    if DRIVE_DEST.exists():
        shutil.rmtree(DRIVE_DEST)
    DRIVE_DEST.mkdir(parents=True)
    total = 0
    for name in names:
        im = Image.open(source / f"{name}.png").convert("RGBA")
        if name.startswith("wheel"):
            im = im.crop((0, row0, im.width, row1))
        out = DRIVE_DEST / f"{name}.webp"
        im.save(out, "WEBP", quality=QUALITY, method=6, exact=True)
        total += out.stat().st_size
    (DRIVE_DEST / "stage.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"wrote {len(names)} photos to {DRIVE_DEST.relative_to(ROOT)} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--drive":
        raise SystemExit(export_drive(Path(args[1]) if len(args) > 1 else DRIVE_SOURCE))
    raise SystemExit(main(Path(args[0]) if args else SOURCE))

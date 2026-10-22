"""Copy a light-stage render into the app's bundled assets, compressed.

    python scripts/hero/export_stage.py [stage_dir]

Reads ``render_top.py --stage`` output (default ``assets_src/sf26/render/stage``: the
``light_<row>_<col>.png`` photos, ``ambient.png`` and ``stage.json``) and writes them to
``src/f1sim/app/assets/hero/stage`` as WebP (quality ``QUALITY``; the ambient photo keeps its
alpha). That's what the menu shows (:mod:`f1sim.app.ui.light_stage`), on every checkout and in
the installed package: the 45 photos of the black car come to under 1 MB.

Re-run it after re-rendering the stage, then commit the folder.
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
QUALITY = 95


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


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE))

"""Download the open map data a circuit's loader asset is baked from (build time only).

    python scripts/geo/fetch.py silverstone              # Aleph if installed, else direct
    python scripts/geo/fetch.py silverstone --direct     # never use Aleph
    python scripts/geo/fetch.py my_track --bbox S W N E  # a circuit not in CIRCUITS

Writes ``assets_src/geo/<track_id>/`` (git-ignored): ``map.osm`` and terrain GeoTIFFs, which
:mod:`bake` reads. Only two sources, both open and fine for a public repo:

- **OpenStreetMap** (ODbL, (c) OpenStreetMap contributors): the ``highway=raceway`` ways.
- **Terrain** (AWS Terrain Tiles, the Tilezen joerd set; sources and their attribution in
  https://github.com/tilezen/joerd/blob/master/docs/attribution.md): EPSG:3857 heights.

**With Aleph** (https://github.com/Belluxx/Aleph, MIT; ``pip install
git+https://github.com/Belluxx/Aleph`` into any venv so ``alephgeo`` is on PATH) this runs
``alephgeo capture create --bbox ... --sources osm terrain --yes --json``: the full OSM map
of the box (from a Geofabrik regional extract) and a merged ``terrain.tif``. Never pass
Aleph's ``satellite``, ``mesh`` or ``streetview`` sources: they are Google data.

**Direct** fetches the same two sources without Aleph: the raceway ways from the Overpass API
(or, when every Overpass server is busy, the box from the main OSM API's ``map`` call) and
the raw ``geotiff/{z}/{x}/{y}.tif`` terrain tiles from the public S3 bucket.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "assets_src" / "geo"
USER_AGENT = "f1sim-geo/0.1 (build-time asset baking; github.com/rrahulnanduri/f1-ml-simulation)"

# (south, west, north, east) in degrees, the circuit plus a few hundred metres of land
CIRCUITS = {
    "silverstone": (52.055, -1.040, 52.090, -0.990),
}

OVERPASS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OSM_MAP = "https://api.openstreetmap.org/api/0.6/map?bbox={w},{s},{e},{n}"
TERRAIN_URL = "https://elevation-tiles-prod.s3.amazonaws.com/geotiff/{z}/{x}/{y}.tif"
TERRAIN_ZOOM = 13  # 9.6 m Mercator pixels, about 6 m on the ground at 52 deg N


def _get(url: str, data: bytes | None = None, timeout: float = 90) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_osm(bbox, out: Path) -> None:
    s, w, n, e = bbox
    query = f'[out:xml][timeout:60];way["highway"="raceway"]({s},{w},{n},{e});(._;>;);out body;'
    body = urllib.parse.urlencode({"data": query}).encode()
    delay = 5.0
    for attempt in range(3):
        for url in OVERPASS:
            try:
                data = _get(url, body, timeout=40)
            except OSError as exc:
                print(f"  {url}: {exc}", file=sys.stderr)
                continue
            if b"<osm" in data[:500] and b"<way" in data:
                out.write_bytes(data)
                print(f"OSM: {out} ({len(data) // 1024} kB, from {url})")
                return
            print(f"  {url}: busy or empty", file=sys.stderr)
        print(f"  retrying in {delay:.0f} s (attempt {attempt + 1})", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    # The main API's map call: everything in the box (raceway_ways filters it). Fine for one
    # small box now and then; it refuses boxes over 0.25 square degrees or 50,000 nodes.
    data = _get(OSM_MAP.format(w=w, s=s, e=e, n=n), timeout=180)
    if b"<osm" not in data[:500]:
        raise SystemExit("Neither Overpass nor the OSM API answered; try again later")
    out.write_bytes(data)
    print(f"OSM: {out} ({len(data) // 1024} kB, from the OSM API)")


def _tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def fetch_terrain(bbox, folder: Path, zoom: int = TERRAIN_ZOOM) -> None:
    s, w, n, e = bbox
    x0, y0 = _tile(n, w, zoom)
    x1, y1 = _tile(s, e, zoom)
    folder.mkdir(parents=True, exist_ok=True)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            path = folder / f"{zoom}_{x}_{y}.tif"
            if not path.is_file():
                path.write_bytes(_get(TERRAIN_URL.format(z=zoom, x=x, y=y)))
    print(f"Terrain: {(x1 - x0 + 1) * (y1 - y0 + 1)} z{zoom} tiles in {folder}")


def fetch_aleph(bbox, out: Path) -> None:
    exe = shutil.which("alephgeo")
    s, w, n, e = bbox
    cmd = [
        exe,
        "capture",
        "create",
        "--bbox",
        str(s),
        str(w),
        str(n),
        str(e),
        "--sources",
        "osm",
        "terrain",
        "--yes",
        "--json",
        "-o",
        str(out / "aleph"),
    ]
    result = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    run = Path(result.get("folder") or result.get("path") or "")
    if not (run / "map.osm").is_file():
        hits = sorted((out / "aleph").rglob("map.osm"))
        if not hits:
            raise SystemExit(f"Aleph wrote no map.osm: {result}")
        run = hits[-1].parent
    shutil.copy2(run / "map.osm", out / "map.osm")
    (out / "terrain").mkdir(exist_ok=True)
    shutil.copy2(run / "terrain.tif", out / "terrain" / "terrain.tif")
    print(f"Aleph capture {run} -> {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("track_id")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"))
    ap.add_argument("--direct", action="store_true", help="fetch without Aleph")
    args = ap.parse_args(argv)
    bbox = tuple(args.bbox) if args.bbox else CIRCUITS.get(args.track_id)
    if bbox is None:
        raise SystemExit(f"No bbox for {args.track_id!r}: pass --bbox S W N E")
    out = OUT_ROOT / args.track_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "bbox.json").write_text(json.dumps({"bbox_swne": bbox}))
    if not args.direct and shutil.which("alephgeo"):
        fetch_aleph(bbox, out)
    else:
        fetch_osm(bbox, out / "map.osm")
        fetch_terrain(bbox, out / "terrain")


if __name__ == "__main__":
    main()

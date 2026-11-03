"""Readers for the two open map sources the loader asset is baked from (build time only).

- **OpenStreetMap** (ODbL): an OSM XML file (Aleph's ``map.osm``, or an Overpass extract).
  :func:`raceway_ways` returns the ``highway=raceway`` ways as lat/lon polylines.
- **Terrain** (AWS Terrain Tiles, the Mapzen/Tilezen open elevation set): GeoTIFFs in EPSG:3857
  metres, Int16 or Float32, deflate-compressed with a predictor. Aleph's merged
  ``terrain.tif`` and the raw ``geotiff/{z}/{x}/{y}.tif`` tiles both read with
  :class:`Terrain`.

Nothing here imports Aleph or any GIS library: numpy, zlib and the standard library only.
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

MERCATOR_R = 6378137.0


def mercator(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 degrees to EPSG:3857 metres."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    x = MERCATOR_R * np.radians(lon)
    y = MERCATOR_R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def inverse_mercator(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(np.asarray(x, dtype=np.float64) / MERCATOR_R)
    y = np.asarray(y, dtype=np.float64)
    lat = np.degrees(2 * np.arctan(np.exp(y / MERCATOR_R)) - np.pi / 2)
    return lat, lon


class LocalFrame:
    """East/north metres about an anchor: EPSG:3857 shifted to the anchor and scaled by
    ``cos(lat0)``, so a metre is a ground metre near the anchor (Mercator's scale error over a
    circuit, a few km, is under 0.1 %)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.mx0, self.my0 = (float(v) for v in mercator(lat0, lon0))
        self.k = math.cos(math.radians(lat0))

    def from_latlon(self, lat, lon) -> np.ndarray:
        mx, my = mercator(lat, lon)
        return np.stack([(mx - self.mx0) * self.k, (my - self.my0) * self.k], axis=-1)

    def to_mercator(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float64)
        return xy[..., 0] / self.k + self.mx0, xy[..., 1] / self.k + self.my0


def raceway_ways(osm_path: Path) -> list[dict]:
    """Every ``highway=raceway`` way in an OSM XML file: ``{"id", "tags", "nodes", "lat", "lon"}``
    (``nodes`` are the node ids, so ways that share an end can be chained)."""
    root = ET.parse(osm_path).getroot()
    coords: dict[str, tuple[float, float]] = {}
    for node in root.iter("node"):
        coords[node.get("id")] = (float(node.get("lat")), float(node.get("lon")))
    ways = []
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        if tags.get("highway") != "raceway":
            continue
        refs = [nd.get("ref") for nd in way.iter("nd") if nd.get("ref") in coords]
        if len(refs) < 2:
            continue
        lat, lon = np.array([coords[r] for r in refs]).T
        ways.append({"id": way.get("id"), "tags": tags, "nodes": refs, "lat": lat, "lon": lon})
    return ways


# --- GeoTIFF ---------------------------------------------------------------------------------

_TYPES = {1: "B", 2: "c", 3: "H", 4: "I", 5: "II", 11: "f", 12: "d", 16: "Q"}


class GeoTIFF:
    """One single-band, tiled or stripped GeoTIFF with a pixel scale and tie point."""

    def __init__(self, path: Path):
        raw = Path(path).read_bytes()
        if raw[:4] not in (b"II*\0", b"MM\0*"):
            raise ValueError(f"{path}: not a classic TIFF")
        o = "<" if raw[:2] == b"II" else ">"
        (ifd,) = struct.unpack(o + "I", raw[4:8])
        (count,) = struct.unpack(o + "H", raw[ifd : ifd + 2])
        tags: dict[int, tuple] = {}
        for i in range(count):
            e = raw[ifd + 2 + 12 * i : ifd + 14 + 12 * i]
            code, kind, n, loc = struct.unpack(o + "HHII", e)
            if kind not in _TYPES:
                continue
            fmt = _TYPES[kind]
            size = struct.calcsize(o + fmt) * n
            data = e[8 : 8 + size] if size <= 4 else raw[loc : loc + size]
            tags[code] = struct.unpack(o + fmt * n, data)
        self.width, self.height = tags[256][0], tags[257][0]
        bits, fmt = tags[258][0], tags.get(339, (1,))[0]
        if (bits, fmt) == (16, 2):
            dtype = np.dtype(o + "i2")
        elif (bits, fmt) == (32, 3):
            dtype = np.dtype(o + "f4")
        else:
            raise ValueError(f"{path}: expected Int16 or Float32 heights, got {bits} bits/{fmt}")
        compression = tags.get(259, (1,))[0]
        predictor = tags.get(317, (1,))[0]
        if 322 in tags:
            bw, bh = tags[322][0], tags[323][0]
            offsets, counts = tags[324], tags[325]
        else:
            bw, bh = self.width, tags.get(278, (self.height,))[0]
            offsets, counts = tags[273], tags[279]
        across = -(-self.width // bw)
        out = np.empty((-(-self.height // bh) * bh, across * bw), dtype=np.float32)
        for k, (off, n) in enumerate(zip(offsets, counts, strict=True)):
            block = raw[off : off + n]
            if compression in (8, 32946):
                block = zlib.decompress(block)
            elif compression != 1:
                raise ValueError(f"{path}: unsupported compression {compression}")
            r0, c0 = (k // across) * bh, (k % across) * bw
            out[r0 : r0 + bh, c0 : c0 + bw] = _unpredict(block, bw, bh, dtype, predictor)
        self.z = out[: self.height, : self.width]
        sx, sy = tags[33550][:2]
        _, _, _, x0, y0, _ = tags[33922][:6]
        self.x0, self.y0, self.sx, self.sy = x0, y0, sx, sy  # upper-left corner, metres/pixel

    def contains(self, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        c = (mx - self.x0) / self.sx
        r = (self.y0 - my) / self.sy
        return (c >= 0) & (c <= self.width) & (r >= 0) & (r <= self.height)

    def sample(self, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        """Bilinear height at EPSG:3857 points (pixel centres at +0.5)."""
        c = np.clip((np.asarray(mx) - self.x0) / self.sx - 0.5, 0, self.width - 1.001)
        r = np.clip((self.y0 - np.asarray(my)) / self.sy - 0.5, 0, self.height - 1.001)
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        fc, fr = c - c0, r - r0
        z = self.z
        top = z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc
        bot = z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc
        return top * (1 - fr) + bot * fr


def _unpredict(block: bytes, bw: int, bh: int, dtype: np.dtype, predictor: int) -> np.ndarray:
    if predictor == 3:  # floating point: byte planes, MSB first, differenced per row
        b = np.frombuffer(block, dtype=np.uint8).reshape(bh, bw * dtype.itemsize)
        b = np.cumsum(b, axis=1, dtype=np.uint8)
        b = b.reshape(bh, dtype.itemsize, bw).transpose(0, 2, 1).copy()
        return b.view(">f4").reshape(bh, bw)
    a = np.frombuffer(block, dtype=dtype).reshape(bh, bw)
    if predictor == 2:
        a = np.cumsum(a, axis=1, dtype=dtype)
    return a.astype(np.float32)


class Terrain:
    """Several GeoTIFFs (tiles or one merged file) sampled as one surface."""

    def __init__(self, paths: list[Path]):
        self.tiles = [GeoTIFF(p) for p in paths]
        if not self.tiles:
            raise ValueError("no terrain GeoTIFFs")

    def sample(self, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        mx, my = np.asarray(mx, dtype=np.float64), np.asarray(my, dtype=np.float64)
        z = np.full(mx.shape, np.nan)
        for t in self.tiles:
            m = t.contains(mx, my) & np.isnan(z)
            if m.any():
                z[m] = t.sample(mx[m], my[m])
        if np.isnan(z).any():
            raise ValueError(f"{int(np.isnan(z).sum())} points fall outside the terrain")
        return z

"""The driving hero: the photographed car lapping a real circuit forever, seen from above.

The car stays fixed on screen, nose up; the circuit moves and turns underneath it. The world
is drawn minimalist: only the track-limit lines and the kerbs.

- :class:`Circuit` is ``scripts/hero/build_circuit.py``'s output: a real qualifying lap's
  racing line every metre with its speed, the track edges and the kerb blocks.
- :class:`Drive` moves the car along the racing line at the lap's real speed and loops at the
  line. The car's reference point is its rear axle (a car turns about it), so its heading is
  the line's direction there. The front wheels steer by the kinematic bicycle angle
  ``atan(wheelbase * curvature)`` of the line under the front axle, times ``STEER_GAIN`` so
  the motion reads at menu size, eased with ``STEER_TAU_S``.
- :func:`world_to_screen` is the transform the track shader uses: rotate the world so the
  car's heading points up, scale, and put the rear axle at its place on screen.

Pure numpy: the menu (:mod:`f1sim.app.menu`) draws it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "hero"
DEFAULT_CIRCUIT = "marina_bay"
STEER_GAIN = 2.5
"""The real steering angle (atan(L / R): about 5 degrees in a 40 m bend) is invisible at
menu size; this exaggerates it, up to ``MAX_STEER_RAD``."""
STEER_TAU_S = 0.12
MAX_STEER_RAD = math.radians(28.0)
PACE = 1.0
"""Playback speed: 1 = the real lap's speed."""


@dataclass(frozen=True)
class Circuit:
    line: np.ndarray  # (N, 2) racing line, one point per metre, closed (no repeat)
    speed: np.ndarray  # (N,) m/s along it
    heading: np.ndarray  # (N,) rad, unwrapped
    curvature: np.ndarray  # (N,) 1/m, left positive
    left: np.ndarray  # (M, 2) left track edge, closed
    right: np.ndarray  # (M, 2) right track edge, closed
    kerb: np.ndarray  # (K, 3, 2) kerb triangles
    kerb_red: np.ndarray  # (K,) True for the red blocks
    name: str
    lap: str  # e.g. "2025 Q RUS"

    @property
    def length(self) -> float:
        return float(self.line.shape[0])  # one point per metre

    @property
    def lap_time(self) -> float:
        """Seconds for one lap at :data:`PACE` 1."""
        return float(np.sum(1.0 / np.maximum(self.speed, 1.0)))


def circuit_path(track_id: str = DEFAULT_CIRCUIT) -> Path:
    return ASSETS / f"{track_id}.npz"


def load_circuit(track_id: str = DEFAULT_CIRCUIT) -> Circuit | None:
    """The bundled hero circuit, or None if it isn't there."""
    path = circuit_path(track_id)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = [str(v) for v in z["meta"]]
        return Circuit(
            line=z["line"].astype(np.float64),
            speed=z["speed"].astype(np.float64),
            heading=np.unwrap(z["heading"].astype(np.float64)),
            curvature=z["curvature"].astype(np.float64),
            left=z["left"].astype(np.float64),
            right=z["right"].astype(np.float64),
            kerb=z["kerb"].astype(np.float64),
            kerb_red=z["kerb_red"].astype(bool),
            name=meta[0],
            lap=meta[1] if len(meta) > 1 else "",
        )


def _sample(values: np.ndarray, s: float, period: float | None = None) -> float:
    """Linear interpolation of per-metre ``values`` at arc length ``s`` (wrapped). With
    ``period``, the values are an unwrapped angle: the wrap adds ``period`` per lap."""
    n = values.shape[0]
    sm = s % n
    i = int(math.floor(sm))
    f = sm - i
    j = (i + 1) % n
    b = values[j] + (period if (period is not None and j == 0) else 0.0)
    return float(values[i] + f * (b - values[i]))


@dataclass
class Drive:
    """The car's progress along a :class:`Circuit` (``s`` = metres, at the rear axle)."""

    circuit: Circuit
    wheelbase_m: float = 3.53
    s: float = 0.0
    steer: float = 0.0
    laps: int = 0

    def __post_init__(self) -> None:
        h = self.circuit.heading
        # The unwrapped heading gains one full turn (or loses one) per lap
        self._turn = float(np.round((h[-1] - h[0]) / (2 * math.pi)) * 2 * math.pi)
        self.steer = self.target_steer()

    @property
    def speed(self) -> float:
        return _sample(self.circuit.speed, self.s)

    def target_steer(self) -> float:
        kappa = _sample(self.circuit.curvature, self.s + self.wheelbase_m)
        delta = math.atan(self.wheelbase_m * kappa) * STEER_GAIN
        return max(-MAX_STEER_RAD, min(MAX_STEER_RAD, delta))

    def update(self, dt: float) -> None:
        dt = max(dt, 0.0)
        n = self.circuit.length
        self.s += self.speed * PACE * dt
        while self.s >= n:
            self.s -= n
            self.laps += 1
        k = 1.0 - math.exp(-dt / STEER_TAU_S) if dt > 0 else 0.0
        self.steer += k * (self.target_steer() - self.steer)

    def pose(self) -> tuple[float, float, float]:
        """``(x, y, heading)`` of the rear axle, world metres and radians."""
        c = self.circuit
        x = _sample(c.line[:, 0], self.s)
        y = _sample(c.line[:, 1], self.s)
        return x, y, _sample(c.heading, self.s, self._turn)


def world_to_screen(points, pose, ppm: float, anchor) -> np.ndarray:
    """World points ``(..., 2)`` -> window pixels: the car's heading up, its rear axle at
    ``anchor`` (window pixels), ``ppm`` pixels per metre. Same maths as the track shader."""
    x, y, heading = pose
    a = math.pi / 2 - heading
    c, s = math.cos(a), math.sin(a)
    d = np.asarray(points, dtype=np.float64) - (x, y)
    rx = c * d[..., 0] - s * d[..., 1]
    ry = s * d[..., 0] + c * d[..., 1]
    return np.stack([anchor[0] + rx * ppm, anchor[1] + ry * ppm], axis=-1)


def edge_quads(edge: np.ndarray, half_width: float) -> np.ndarray:
    """Triangles ``(2M, 3, 2)`` of a closed line ``half_width`` either side of ``edge``."""
    nxt = np.roll(edge, -1, axis=0)
    prv = np.roll(edge, 1, axis=0)
    t = nxt - prv
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1) * half_width
    a, b = edge + n, edge - n
    a2, b2 = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
    t1 = np.stack([a, b, a2], axis=1)
    t2 = np.stack([b, b2, a2], axis=1)
    out = np.empty((2 * edge.shape[0], 3, 2))
    out[0::2], out[1::2] = t1, t2
    return out


# ------------------------------------------------------------------------- the photos


@dataclass(frozen=True)
class DriveStage:
    """The driving light stage (``render_top.py --drive``, exported by ``export_stage.py``)."""

    body: np.ndarray  # (L, H, W, 4) uint8: the car without its front wheels, per softbox
    body_ambient: np.ndarray  # (H, W, 4): at rest, with the soft ground shadow in alpha
    wheels: np.ndarray  # (L, h, w, 4): only the front wheels (a strip of rows around them)
    wheels_ambient: np.ndarray  # (h, w, 4)
    wheel_rows: tuple[int, int]  # the strip's first row and row count in the full image
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    px_per_m: float
    wheel_centres: tuple[tuple[float, float], ...]  # map metres (x right, y down from top)
    wheel_radius: float
    rear_axle_y: float  # map metres
    wheelbase: float

    @property
    def size_m(self) -> tuple[float, float]:
        h, w = self.body_ambient.shape[:2]
        return w / self.px_per_m, h / self.px_per_m


DRIVE_DIR = ASSETS / "drive"


def load_stage(directory: Path | None = None) -> DriveStage | None:
    """The bundled driving stage, or None if it's missing or incomplete."""
    from PIL import Image

    d = Path(directory) if directory is not None else DRIVE_DIR
    info_path = d / "stage.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    xs, ys = tuple(info["xs"]), tuple(info["ys"])
    names = [f"{r}_{c}" for r in range(len(ys)) for c in range(len(xs))]

    def read(name):
        p = d / f"{name}.webp"
        return np.asarray(Image.open(p).convert("RGBA")) if p.is_file() else None

    body = [read(f"body_{n}") for n in names]
    wheels = [read(f"wheel_{n}") for n in names]
    body_amb, wheel_amb = read("body_ambient"), read("wheel_ambient")
    if any(a is None for a in [*body, *wheels, body_amb, wheel_amb]):
        return None
    wheels_info = info["wheels"]
    return DriveStage(
        body=np.stack(body),
        body_ambient=body_amb,
        wheels=np.stack(wheels),
        wheels_ambient=wheel_amb,
        wheel_rows=tuple(info["wheel_rows"]),
        xs=xs,
        ys=ys,
        px_per_m=float(info["px_per_m"]),
        wheel_centres=tuple((float(w["x"]), float(w["y"])) for w in wheels_info),
        wheel_radius=float(max(w["r"] for w in wheels_info)),
        rear_axle_y=float(info["rear_axle_y"]),
        wheelbase=float(info["rear_axle_y"] - info["front_axle_y"]),
    )

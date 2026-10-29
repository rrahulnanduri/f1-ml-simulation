"""The driving hero: the photographed car lapping a real circuit forever, seen from above.

The camera is a fixed top-down view that follows the car without turning: the car drives and
turns on screen, the circuit slides underneath it. The world
is drawn minimalist: only the track-limit lines and the kerbs. Slower cars share the track and
the hero keeps overtaking them.

- :class:`Circuit` is ``scripts/hero/build_circuit.py``'s output: a real qualifying lap's
  racing line every metre with its speed, the track edges and the kerb blocks.
  :func:`track_room` measures the free room either side of the line.
- :class:`Drive` moves a car along the racing line at ``pace`` times the lap's real speed and
  loops at the line. It can run ``d`` metres left of the line (``d`` eases towards
  ``d_target`` like a critically damped spring), so its heading adds the slope
  ``atan(dd/ds)``. The car's reference point is its rear axle (a car turns about it). The
  front wheels steer by the kinematic bicycle angle ``atan(wheelbase * curvature)`` of the
  shifted path under the front axle, times ``STEER_GAIN`` so the motion reads at menu size,
  eased with ``STEER_TAU_S``.
- :class:`Traffic` is the hero and ``N_RIVALS`` slower cars in other finishes. Whenever one is
  within ``PASS_START_M`` ahead, the hero moves over (to the inside of the next corner, or the
  side with more room) and passes; the slower car leaves it room. A car left well behind and
  out of view reappears out of view ahead, so the overtakes never end. Seeded: it repeats.
- :func:`world_to_screen` is the transform the track shader uses: a fixed top-down view
  (world ``VIEW_HEADING`` up, the world never turns), scale, and put the rear axle at its place
  on screen.

Pure numpy: the menu (:mod:`f1sim.app.menu`) draws it.
"""

from __future__ import annotations

import hashlib
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
VIEW_HEADING = math.pi / 2
"""The world direction (rad) that points up on screen. Fixed: the camera follows the hero but
never turns with it."""
PACE = 1.0
"""Playback speed: 1 = the real lap's speed."""
OFFSET_OMEGA = 2.6
"""Natural frequency (rad/s) of the critically damped spring that eases a car's offset from
the racing line: a 2.6 m move takes about 1.5 s."""
CAR_HALF_WIDTH_M = 1.0
CAR_LENGTH_M = 5.6
EDGE_MARGIN_M = 0.3
"""An offset car keeps its side this far inside the track limit."""

N_RIVALS = 5
RIVAL_CLOSING_MS = (6.0, 18.0)
"""How much slower than the real lap's speed (m/s) each slower car drives, drawn again every
time it reappears. A fixed gap in speed, not a share of it, so the hero catches a car at the
same rate on a straight as in a corner and the overtakes don't bunch up in the slow parts."""
RIVAL_MIN_SHARE = 0.5
"""...but never below this share of the real speed (the car would crawl in a slow corner)."""
RIVAL_WOBBLE_M = 0.35
RESPAWN_BEHIND_M = 25.0
RESPAWN_AHEAD_M = (60.0, 320.0)
RIVAL_SPACING_M = 30.0
"""Cars more than ``RESPAWN_BEHIND_M`` behind the hero (and out of view) reappear this far
ahead (out of view too), at least ``RIVAL_SPACING_M`` from every other car."""
VIEW_M = 15.0
"""Radius around the hero that the window shows (the menu sets it from its layout)."""
PASS_START_M = 35.0
PASS_CLEAR_M = 8.0
PASS_OFFSET_M = 2.6
PASS_SEPARATION_M = 2 * CAR_HALF_WIDTH_M + 0.9
PASS_SPARE_M = 0.2
"""The hero moves up to ``PASS_OFFSET_M`` off the line to pass a car within ``PASS_START_M``
ahead, and back once ``PASS_CLEAR_M`` clear. The car being passed moves the other way, enough
for ``PASS_SEPARATION_M`` between their centre lines; a side without that (and
``PASS_SPARE_M``) is too tight."""
PASS_COMMIT_M = 15.0
"""Until the nearest car is this close, the hero may still switch to the other side."""
PASS_LOOK_M = 100.0
"""How far ahead the hero looks for the next corner (whose inside it takes)."""
GIVE_ROOM_M = 0.5
GIVE_MAX_M = 1.3
"""A car being passed moves this far the other way, more (up to ``GIVE_MAX_M``) where the
hero can't move ``PASS_OFFSET_M`` over. If that still isn't enough, the hero waits behind."""
HERO_FOLLOW_M = CAR_LENGTH_M + 1.5
FOLLOW_M = CAR_LENGTH_M + 3.0
FOLLOW_REACH_M = 3.5
FOLLOW_GAIN = 2.0
"""A car within ``FOLLOW_REACH_M`` of its following distance (rear axle to rear axle) behind
one it can't pass matches its pace, plus ``FOLLOW_GAIN`` m/s per metre of spare gap."""
SETTLE_S = 1.3
"""The offset spring's settling time: limits are checked this far ahead (at the car's
speed), so a car is back in before the track narrows."""


@dataclass(frozen=True)
class Finish:
    """A body finish. The photos are of the hero's black: ``albedo_scale`` scales the
    bodywork's colour from it; the rest replace the bodywork's material."""

    name: str
    albedo_scale: float
    roughness: float
    metallic: float
    coat: float


HERO_FINISH = Finish("metallic black", 1.0, 0.30, 0.85, 1.0)
RIVAL_FINISHES = (
    Finish("gunmetal", 6.0, 0.30, 0.90, 1.0),
    Finish("graphite", 3.5, 0.42, 0.35, 0.8),
    Finish("satin grey", 8.0, 0.55, 0.0, 0.25),
)


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


def track_room(circuit: Circuit) -> tuple[np.ndarray, np.ndarray]:
    """Free room ``(left, right)`` per metre of the racing line: the distance to each track
    limit, metres."""
    import shapely

    points = shapely.points(circuit.line)
    return tuple(
        shapely.distance(points, shapely.linearrings(edge))
        for edge in (circuit.left, circuit.right)
    )


def _window(values: np.ndarray, s: float, ahead: float) -> np.ndarray:
    """``values`` (per metre, closed) from ``s`` to ``s + ahead``."""
    n = values.shape[0]
    i = np.arange(int(math.floor(s)), int(math.ceil(s + ahead)) + 1) % n
    return values[i]


@dataclass
class Drive:
    """A car's progress along a :class:`Circuit`: ``s`` = metres along the racing line, at
    the rear axle; ``d`` = metres left of the line (``vd`` its rate, m/s)."""

    circuit: Circuit
    wheelbase_m: float = 3.53
    s: float = 0.0
    steer: float = 0.0
    laps: int = 0
    pace: float = PACE
    slow_ms: float = 0.0
    """Metres per second below ``pace`` times the lap's speed (``RIVAL_MIN_SHARE`` at least)."""
    d: float = 0.0
    vd: float = 0.0
    d_target: float = 0.0
    rate_cap: float | None = None
    """Upper limit on :attr:`rate` for the next update (following a slower car), m/s."""

    def __post_init__(self) -> None:
        h = self.circuit.heading
        # The unwrapped heading gains one full turn (or loses one) per lap
        self._turn = float(np.round((h[-1] - h[0]) / (2 * math.pi)) * 2 * math.pi)
        self._ad = 0.0  # lateral acceleration of the offset, m/s^2
        self.steer = self.target_steer()

    @property
    def speed(self) -> float:
        """Speed along the car's own path, m/s."""
        v = self.natural_speed(self.s)
        if self.rate_cap is not None:
            v = min(v, self.rate_cap * self._stretch())
        return v

    def natural_speed(self, s: float) -> float:
        """The speed at ``s`` with no car ahead, m/s."""
        v = _sample(self.circuit.speed, s)
        return max(v * self.pace - self.slow_ms, RIVAL_MIN_SHARE * v * self.pace)

    def _stretch(self) -> float:
        """Metres of the car's path per metre of the line: an offset path is longer outside
        a bend, shorter inside it."""
        return max(1.0 - _sample(self.circuit.curvature, self.s) * self.d, 0.3)

    @property
    def rate(self) -> float:
        """Progress along the racing line, m/s."""
        return self.speed / self._stretch()

    def path_curvature(self, s: float) -> float:
        """Curvature (1/m, left positive) of the car's path at ``s``: the racing line's,
        shifted by the offset, plus the bend the offset's own change adds."""
        kappa = _sample(self.circuit.curvature, s)
        shifted = kappa / max(1.0 - kappa * self.d, 0.3)
        return shifted + self._ad / max(self.speed, 1.0) ** 2

    def target_steer(self) -> float:
        kappa = self.path_curvature(self.s + self.wheelbase_m)
        delta = math.atan(self.wheelbase_m * kappa) * STEER_GAIN
        return max(-MAX_STEER_RAD, min(MAX_STEER_RAD, delta))

    def place(self, s: float, d: float = 0.0) -> None:
        """Put the car at ``s``, ``d`` at rest laterally (for a car out of view)."""
        self.s = s % self.circuit.length
        self.d = self.d_target = d
        self.vd = self._ad = 0.0
        self.steer = self.target_steer()

    def update(self, dt: float) -> None:
        dt = max(dt, 0.0)
        n = self.circuit.length
        # The offset: a critically damped spring towards its target (exact for a step dt)
        w = OFFSET_OMEGA
        e0, v0 = self.d - self.d_target, self.vd
        decay = math.exp(-w * dt)
        b = v0 + w * e0
        e1 = (e0 + b * dt) * decay
        v1 = (b - w * (e0 + b * dt)) * decay
        self._ad = (v1 - v0) / dt if dt > 0 else 0.0
        self.d, self.vd = self.d_target + e1, v1
        self.s += self.rate * dt
        while self.s >= n:
            self.s -= n
            self.laps += 1
        k = 1.0 - math.exp(-dt / STEER_TAU_S) if dt > 0 else 0.0
        self.steer += k * (self.target_steer() - self.steer)

    def pose(self) -> tuple[float, float, float]:
        """``(x, y, heading)`` of the rear axle, world metres and radians."""
        c = self.circuit
        h = _sample(c.heading, self.s, self._turn)
        x = _sample(c.line[:, 0], self.s) - math.sin(h) * self.d
        y = _sample(c.line[:, 1], self.s) + math.cos(h) * self.d
        return x, y, h + math.atan2(self.vd, max(self.speed, 1.0))


def footprint(pose: tuple[float, float, float]) -> np.ndarray:
    """The four corners ``(4, 2)`` of a car's plan rectangle, from its rear-axle pose."""
    x, y, h = pose
    f = np.array([math.cos(h), math.sin(h)])
    left = np.array([-f[1], f[0]])
    back = np.array([x, y]) - 1.0 * f  # the rear wing overhangs the axle
    return np.array(
        [
            back + sx * CAR_LENGTH_M * f + sy * CAR_HALF_WIDTH_M * left
            for sx, sy in ((0, -1), (1, -1), (1, 1), (0, 1))
        ]
    )


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two convex footprints (corners in order) intersect (separating axes)."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for nx, ny in np.stack([-edges[:, 1], edges[:, 0]], axis=1):
            pa, pb = a @ (nx, ny), b @ (nx, ny)
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


@dataclass
class Rival:
    """A slower car: its :class:`Drive`, finish and wobble about the line."""

    drive: Drive
    finish: Finish
    wobble_phase: float
    wobble_rate: float  # rad/s
    respawns: int = 0


class Traffic:
    """The hero and the slower cars it keeps passing (see the module docstring)."""

    def __init__(
        self,
        hero: Drive,
        n_rivals: int = N_RIVALS,
        seed: int = 0,
        view_m: float = VIEW_M,
        room: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.hero = hero
        self.circuit = hero.circuit
        self.view_m = view_m
        self.rng = np.random.default_rng(seed)
        self.room_left, self.room_right = room if room is not None else track_room(self.circuit)
        self.t = 0.0
        self.side = 0.0  # +1 passing on the left, -1 on the right
        self.rivals: list[Rival] = []
        self.respawn_log: list[tuple[float, float, float]] = []
        """``(time, distance from the hero before, after)`` of every respawn, metres."""
        for i in range(n_rivals):
            drive = Drive(
                self.circuit,
                wheelbase_m=hero.wheelbase_m,
                slow_ms=float(self.rng.uniform(*RIVAL_CLOSING_MS)),
            )
            rival = Rival(
                drive,
                RIVAL_FINISHES[i % len(RIVAL_FINISHES)],
                float(self.rng.uniform(0, 2 * math.pi)),
                float(self.rng.uniform(0.25, 0.6)),
            )
            drive.place(self._spawn_s(drive.slow_ms), 0.0)
            self.rivals.append(rival)

    # ---------------------------------------------------------------- geometry

    def gap(self, rival: Rival) -> float:
        """Metres ``rival`` is ahead of the hero along the line (negative: behind)."""
        n = self.circuit.length
        return (rival.drive.s - self.hero.s + 0.5 * n) % n - 0.5 * n

    def distance(self, rival: Rival) -> float:
        """Straight-line distance from the hero, metres."""
        hx, hy, _ = self.hero.pose()
        x, y, _ = rival.drive.pose()
        return math.hypot(x - hx, y - hy)

    def max_offset(self, s: float, side: float, ahead: float) -> float:
        """The largest offset to ``side`` that keeps a car inside the limits from ``s`` to
        ``s + ahead``."""
        room = self.room_left if side > 0 else self.room_right
        return float(np.min(_window(room, s, ahead))) - CAR_HALF_WIDTH_M - EDGE_MARGIN_M

    def _spawn_s(self, slow_ms: float, moving: Rival | None = None) -> float:
        """A place ahead of the hero, out of view, clear of the other cars (but ``moving``),
        for a car that will drive ``slow_ms`` below the lap speed."""
        n = self.circuit.length
        hx, hy, _ = self.hero.pose()
        others = [r.drive.s for r in self.rivals if r is not moving]
        best, best_clear = None, -1.0
        for _ in range(40):
            lo, hi = RESPAWN_AHEAD_M[0], min(RESPAWN_AHEAD_M[1], 0.4 * n)
            s = (self.hero.s + float(self.rng.uniform(lo, hi))) % n
            clear = min(
                (abs((s - o + 0.5 * n) % n - 0.5 * n) for o in others), default=float("inf")
            )
            x, y = _sample(self.circuit.line[:, 0], s), _sample(self.circuit.line[:, 1], s)
            if math.hypot(x - hx, y - hy) <= self.view_m:
                continue
            if clear >= RIVAL_SPACING_M:
                return s
            if clear > best_clear:
                best, best_clear = s, clear
        return best if best is not None else (self.hero.s + RESPAWN_AHEAD_M[1]) % n

    # ------------------------------------------------------------------ update

    def _pass_reach(self, gap: float, rival: Rival) -> float:
        """Metres of track the hero covers passing ``rival`` from ``gap`` ahead until it's
        ``PASS_CLEAR_M`` clear, plus the offset spring's settling distance."""
        v = self.hero.natural_speed(self.hero.s)  # the natural speeds: no flicker
        closing = max(v - rival.drive.natural_speed(rival.drive.s), 3.0)
        return (max(gap, 0.0) + PASS_CLEAR_M + CAR_LENGTH_M) * v / closing + v * SETTLE_S

    def _pass_room(self, side: float, gap: float, rival: Rival) -> float:
        """Sideways separation the hero and ``rival``, ``gap`` ahead, could reach, the hero
        passing on ``side``."""
        reach = self._pass_reach(gap, rival)
        hero = min(PASS_OFFSET_M, self.max_offset(self.hero.s, side, reach))
        other = self.max_offset(rival.drive.s, -side, reach - max(gap, 0.0))
        return max(hero, 0.0) + min(max(other, 0.0), GIVE_MAX_M)

    def _choose_side(self, gap: float, rival: Rival) -> float:
        """The inside of the next corner, or the side with more room if that's too tight."""
        kappa = _window(self.circuit.curvature, self.hero.s, gap + PASS_LOOK_M)
        inside = 1.0 if kappa[np.argmax(np.abs(kappa))] >= 0 else -1.0
        room_in = self._pass_room(inside, gap, rival)
        if room_in >= PASS_SEPARATION_M + PASS_SPARE_M:
            return inside
        return inside if room_in >= self._pass_room(-inside, gap, rival) else -inside

    def passing(self) -> list[Rival]:
        """The cars being passed: from ``PASS_START_M`` ahead to ``PASS_CLEAR_M`` behind."""
        return [r for r in self.rivals if -PASS_CLEAR_M < self.gap(r) < PASS_START_M]

    def update(self, dt: float) -> None:
        dt = max(dt, 0.0)
        self.t += dt
        hero = self.hero
        n = self.circuit.length
        # Respawn cars left behind, out of view
        for r in self.rivals:
            if self.gap(r) < -RESPAWN_BEHIND_M and self.distance(r) > self.view_m:
                before = self.distance(r)
                r.drive.slow_ms = float(self.rng.uniform(*RIVAL_CLOSING_MS))
                r.drive.place(self._spawn_s(r.drive.slow_ms, r), 0.0)
                r.respawns += 1
                self.respawn_log.append((self.t, before, self.distance(r)))
        # The pass: a side is picked when a car comes within reach, and held until every car
        # in reach is passed
        passing = self.passing()
        if not passing:
            self.side = 0.0
        else:
            gap, nearest = min((self.gap(r), i) for i, r in enumerate(passing))
            rival = passing[nearest]
            # Pick a side; change it while the car is still well ahead if it's too tight
            if not self.side or (
                gap > PASS_COMMIT_M and self._pass_room(self.side, gap, rival) < PASS_SEPARATION_M
            ):
                self.side = self._choose_side(gap, rival)
        give = 0.0
        if self.side:
            reach = max(self._pass_reach(self.gap(r), r) for r in passing)
            room = self.max_offset(hero.s, self.side, reach)
            if any(abs(self.gap(r) - 1.0) < CAR_LENGTH_M + 1.0 for r in passing):
                # Alongside: hold the offset as long as the track right ahead allows it
                now = self.max_offset(hero.s, self.side, hero.speed * SETTLE_S)
                room = max(room, min(abs(hero.d_target), now))
            off = max(min(PASS_OFFSET_M, room), 0.0)
            hero.d_target = self.side * off
            give = min(max(GIVE_ROOM_M, PASS_SEPARATION_M - off), GIVE_MAX_M)
        else:
            hero.d_target = 0.0
        # The slower cars wobble about the line; any being passed leaves room
        for r in self.rivals:
            dr = r.drive
            if self.side and r in passing:
                want = -self.side * give
            else:
                want = RIVAL_WOBBLE_M * math.sin(r.wobble_phase + r.wobble_rate * self.t)
            side = 1.0 if want >= 0 else -1.0
            limit = self.max_offset(dr.s, side, dr.speed * SETTLE_S)
            dr.d_target = side * min(abs(want), max(limit, 0.0))
        # Nobody drives into a car ahead: each car follows any it isn't (or won't stay) clear
        # of sideways. Front to back, so each sees its leader's rate.
        cars = [hero, *(r.drive for r in self.rivals)]
        cars.sort(key=lambda car: -((car.s - hero.s + 0.5 * n) % n))
        for car in cars:
            car.rate_cap = None
            close = HERO_FOLLOW_M if car is hero else FOLLOW_M
            for other in cars:
                g = (other.s - car.s + 0.5 * n) % n - 0.5 * n
                if other is car or not 0.0 < g < close + FOLLOW_REACH_M:
                    continue
                apart = min(abs(car.d - other.d), abs(car.d_target - other.d_target))
                if apart < PASS_SEPARATION_M - 0.1:
                    cap = max(other.rate + FOLLOW_GAIN * (g - close), 0.0)
                    car.rate_cap = min(car.rate_cap or math.inf, cap)
        for car in cars:
            car.update(dt)

    def cars(self) -> list[tuple[Drive, Finish]]:
        """Every slower car with its finish (the hero is :attr:`hero`, in ``HERO_FINISH``)."""
        return [(r.drive, r.finish) for r in self.rivals]


def world_to_screen(points, pose, ppm: float, anchor) -> np.ndarray:
    """World points ``(..., 2)`` -> window pixels: a fixed top-down view (world ``VIEW_HEADING``
    up), the car's rear axle at ``anchor`` (window pixels), ``ppm`` pixels per metre. Same maths
    as the track shader."""
    x, y, _ = pose
    a = math.pi / 2 - VIEW_HEADING
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
    """The driving light stage (``render_top.py --drive``, exported by ``export_stage.py``).

    Two layers, the body (everything but the front wheels) and the front wheels (a strip of
    rows around them, ``wheel_rows``), each with:

    - ``light``: per softbox position, the light falling on the car (grey, uint8, sRGB
      encoded; a white floor under the softbox reads ``white_floor`` linear), at half the
      maps' resolution: the menu only takes the shadowing from it (a quarter of the memory
      and upload);
    - ``ambient``: the same at rest (RGBA; the body's alpha adds the soft ground shadow),
      ``ambient_gain`` times brighter;
    - the data maps ``albedo`` (RGBA, alpha = coverage), ``normal``, ``surface`` (roughness,
      metallic, height / ``height_scale``) and ``ao`` (occlusion, clear-coat weight).
    """

    body: dict[str, np.ndarray]
    wheels: dict[str, np.ndarray]
    wheel_rows: tuple[int, int]  # the strip's first row and row count in the full image
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    px_per_m: float
    wheel_centres: tuple[tuple[float, float], ...]  # map metres (x right, y down from top)
    wheel_radius: float
    rear_axle_y: float  # map metres
    wheelbase: float
    light_z: float
    softbox_m: float
    white_floor: float
    ambient_gain: float
    height_scale: float

    @property
    def size_m(self) -> tuple[float, float]:
        h, w = self.body["albedo"].shape[:2]
        return w / self.px_per_m, h / self.px_per_m


DRIVE_DIR = ASSETS / "drive"
STAGE_MAPS = ("ambient", "albedo", "normal", "surface", "ao")


STAGE_CACHE_VERSION = 1


def stage_cache_path(directory: Path) -> Path:
    """Where the decoded arrays of a driving stage are cached (the user cache, never the
    repo)."""
    from f1sim import paths

    key = hashlib.sha1(str(Path(directory).resolve()).encode("utf-8")).hexdigest()[:12]
    return paths.cache_dir() / "hero" / f"drive_{key}.npz"


def _decode_stage(d: Path, cells: list[str]) -> dict | None:
    """Both layers' arrays from the WebP files (about 2 s: hence the cache)."""
    from PIL import Image

    def read(name, mode, half=False):
        p = d / f"{name}.webp"
        if not p.is_file():
            return None
        im = Image.open(p).convert(mode)
        if half:  # box-filtered to half size, covering the same area
            im = im.resize(((im.width + 1) // 2, (im.height + 1) // 2), Image.BOX)
        return np.asarray(im)

    layers = {}
    for layer in ("body", "wheel"):
        light = [read(f"{layer}_{c}", "L", half=True) for c in cells]
        maps = {m: read(f"{layer}_{m}", "RGBA") for m in STAGE_MAPS}
        if any(a is None for a in [*light, *maps.values()]):
            return None
        layers[layer] = {"light": np.stack(light), **maps}
    return layers


def load_stage(directory: Path | None = None) -> DriveStage | None:
    """The bundled driving stage, or None if it's missing or incomplete. The decoded arrays
    are cached (:func:`stage_cache_path`) while the files are unchanged."""
    d = Path(directory) if directory is not None else DRIVE_DIR
    info_path = d / "stage.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if "white_floor" not in info:
        return None  # an older set, before the live light model
    xs, ys = tuple(info["xs"]), tuple(info["ys"])
    cells = [f"{r}_{c}" for r in range(len(ys)) for c in range(len(xs))]
    stamp = max(p.stat().st_mtime for p in d.iterdir())
    cache = stage_cache_path(d)
    layers = None
    if cache.is_file() and cache.stat().st_mtime >= stamp:
        try:
            with np.load(cache, allow_pickle=False) as z:
                if int(z["version"]) == STAGE_CACHE_VERSION:
                    layers = {
                        layer: {k: z[f"{layer}_{k}"] for k in ("light", *STAGE_MAPS)}
                        for layer in ("body", "wheel")
                    }
        except (OSError, ValueError, KeyError):
            layers = None
    if layers is None:
        layers = _decode_stage(d, cells)
        if layers is None:
            return None
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            flat = {f"{layer}_{k}": v for layer, arrs in layers.items() for k, v in arrs.items()}
            np.savez(cache, version=STAGE_CACHE_VERSION, **flat)
        except OSError:
            pass
    wheels_info = info["wheels"]
    return DriveStage(
        body=layers["body"],
        wheels=layers["wheel"],
        wheel_rows=tuple(info["wheel_rows"]),
        xs=xs,
        ys=ys,
        px_per_m=float(info["px_per_m"]),
        wheel_centres=tuple((float(w["x"]), float(w["y"])) for w in wheels_info),
        wheel_radius=float(max(w["r"] for w in wheels_info)),
        rear_axle_y=float(info["rear_axle_y"]),
        wheelbase=float(info["rear_axle_y"] - info["front_axle_y"]),
        light_z=float(info["light_z_m"]),
        softbox_m=float(info["softbox_m"]),
        white_floor=float(info["white_floor"]),
        ambient_gain=float(info["ambient_gain"]),
        height_scale=float(info["height_scale_m"]),
    )


# ------------------------------------------------------------------- trackside lamps

LAMP_GAP_M = (23.0, 108.0)
LAMP_OUTSIDE_M = (0.6, 4.5)
"""Lamp units are scattered: each a random gap after the last along the line, on a random
side, a random distance outside the track limit."""
LAMP_RATE = (0.8, 1.25)
"""Each unit flashes at its own rate (times the cycle's)."""
LAMP_HEIGHT_M = 0.7
LAMP_HEADS_M = (0.6, 1.5)
"""Each unit has a red and a blue head, this far apart along the track."""
LAMP_CYCLE_S = 1.0
LAMP_FLASH = ((0.00, "red"), (0.2, "red"), (0.5, "blue"), (0.7, "blue"))
"""When in each cycle a head flashes (seconds): a double strobe, red then blue."""
LAMP_DECAY_S = 0.11
"""Each flash fades this fast: slow enough that its light lingers on the passing cars."""
LAMP_COLOURS = {"red": (1.0, 0.06, 0.04), "blue": (0.10, 0.25, 1.0)}


@dataclass(frozen=True)
class Lamp:
    x: float  # world metres
    y: float
    colour: str  # "red" or "blue"
    phase: float  # seconds into the cycle at t = 0
    rate: float = 1.0  # cycles per LAMP_CYCLE_S


def trackside_lamps(circuit: Circuit, room=None, seed: int = 0) -> list[Lamp]:
    """Flashing lamp units scattered by the track (seeded: see ``LAMP_GAP_M``), a red and a
    blue head each, each unit at its own rate and phase."""
    left_room, right_room = room if room is not None else track_room(circuit)
    rng = np.random.default_rng(seed)
    out = []
    n = int(circuit.length)
    s0 = float(rng.uniform(*LAMP_GAP_M))
    while s0 < n - 0.5 * LAMP_GAP_M[0]:
        i = int(s0) % n
        side = float(rng.choice((-1.0, 1.0)))
        dist = (left_room[i] if side > 0 else right_room[i]) + rng.uniform(*LAMP_OUTSIDE_M)
        phase = float(rng.uniform(0.0, LAMP_CYCLE_S))
        rate = float(rng.uniform(*LAMP_RATE))
        heads = float(rng.uniform(*LAMP_HEADS_M))
        for k, colour in enumerate(("red", "blue")):
            s = s0 + (k - 0.5) * heads
            h = _sample(circuit.heading, s)
            x = _sample(circuit.line[:, 0], s) - math.sin(h) * side * dist
            y = _sample(circuit.line[:, 1], s) + math.cos(h) * side * dist
            out.append(Lamp(x, y, colour, phase, rate))
        s0 += float(rng.uniform(*LAMP_GAP_M))
    return out


def lamp_brightness(lamp: Lamp, t: float) -> float:
    """0..1: a sharp rise at each of the head's flashes, then a fast decay."""
    tc = (t * lamp.rate + lamp.phase) % LAMP_CYCLE_S
    b = 0.0
    for start, colour in LAMP_FLASH:
        if colour != lamp.colour:
            continue
        dt = (tc - start) % LAMP_CYCLE_S
        b = max(b, math.exp(-dt / LAMP_DECAY_S) if dt < 0.5 * LAMP_CYCLE_S else 0.0)
    return b

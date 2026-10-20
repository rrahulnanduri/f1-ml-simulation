"""Vectorised F1 car model: kinematic bicycle with a speed-dependent friction circle.

All quantities are SI (m, s, rad, m/s^2) and every array has shape ``(N,)``, one entry per car.
The model follows docs/revamp-plan.md section 6; :func:`step` documents the exact update rule.

Sign conventions: positive steering (``steer_cmd > 0``, ``delta > 0``) turns the car left, i.e.
yaw increases (counter-clockwise), matching ``curvature > 0 = turning left`` on the track.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

G = 9.81
"""Standard gravity used for every ``*_g`` parameter, m/s^2."""

V_FLOOR = 1.0
"""Speed floor (m/s) in the power-limited drive term ``P / (m * max(v, V_FLOOR))``.

It only stops a division by zero at a standing start: at 1 m/s the power term is ~940 m/s^2,
far above the traction cap ``max_drive_g * g``, so the floor never changes the result.
"""


@dataclass(frozen=True)
class CarParams:
    """Car parameters; defaults are the revamp-plan section 6 table."""

    length_m: float = 5.6
    width_m: float = 2.0
    wheelbase_m: float = 3.6
    max_steer_rad: float = 0.30
    steer_rate_rad_s: float = 1.5
    power_w: float = 750e3
    mass_kg: float = 800.0
    max_drive_g: float = 1.3
    drag_coeff: float = 0.0012
    grip_base_g: float = 1.9
    grip_aero_coeff: float = 0.00069
    max_brake_g: float = 3.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"CarParams.{f.name} must be finite and >= 0, got {value!r}")
        for name in ("wheelbase_m", "mass_kg", "grip_base_g"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"CarParams.{name} must be > 0")


def load_car_params(path: str | Path) -> CarParams:
    """Load :class:`CarParams` from a flat TOML file.

    Keys are ``CarParams`` field names; missing keys keep their defaults. Unknown keys and
    non-numeric values raise ``ValueError``.
    """
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    known = {f.name for f in fields(CarParams)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{path}: unknown car parameter(s): {', '.join(unknown)}")
    values: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path}: {key} must be a number, got {value!r}")
        values[key] = float(value)
    return CarParams(**values)


@dataclass
class CarState:
    """Mutable state of N cars; every field is a float64 array of shape ``(N,)``.

    ``x``, ``y`` position (m), ``yaw`` heading (rad, wrapped to [-pi, pi)), ``v`` speed (m/s,
    always >= 0), ``delta`` front-wheel steering angle (rad, positive = left).
    """

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    delta: np.ndarray

    def __post_init__(self) -> None:
        arrays = {}
        for f in fields(self):
            arr = np.array(getattr(self, f.name), dtype=np.float64, ndmin=1)
            if arr.ndim != 1:
                raise ValueError(f"CarState.{f.name} must be 1-D, got shape {arr.shape}")
            arrays[f.name] = arr
        n = arrays["x"].shape[0]
        for name, arr in arrays.items():
            if arr.shape[0] != n:
                raise ValueError(f"CarState.{name} has length {arr.shape[0]}, expected {n}")
            setattr(self, name, arr)

    @property
    def n(self) -> int:
        """Number of cars."""
        return int(self.x.shape[0])

    @classmethod
    def at(cls, n: int, x: float, y: float, yaw: float, v: float = 0.0) -> CarState:
        """N identical cars at ``(x, y)`` heading ``yaw`` at speed ``v``, wheels straight."""
        return cls(
            x=np.full(n, x, dtype=np.float64),
            y=np.full(n, y, dtype=np.float64),
            yaw=np.full(n, yaw, dtype=np.float64),
            v=np.full(n, v, dtype=np.float64),
            delta=np.zeros(n, dtype=np.float64),
        )


@dataclass(frozen=True)
class StepInfo:
    """What one :func:`step` actually applied, per car.

    ``a_long``: realised longitudinal acceleration ``(v_new - v) / dt`` (m/s^2), including drag,
    slide scrub and the ``v >= 0`` clamp. ``a_lat``: realised lateral (centripetal) acceleration
    ``v_new * yaw_rate`` (m/s^2, positive = left). ``sliding``: bool, a grip limit was exceeded.
    """

    a_long: np.ndarray
    a_lat: np.ndarray
    sliding: np.ndarray


def grip_limit(v, params: CarParams) -> np.ndarray:
    """Friction-circle radius ``a_max(v) = g * (grip_base_g + grip_aero_coeff * v^2)``, m/s^2."""
    v = np.asarray(v, dtype=np.float64)
    return G * (params.grip_base_g + params.grip_aero_coeff * v * v)


def step(state: CarState, steer_cmd, pedal_cmd, params: CarParams, dt: float) -> StepInfo:
    """Advance all cars by ``dt`` seconds, updating ``state`` in place.

    ``steer_cmd`` and ``pedal_cmd`` are scalars or ``(N,)`` arrays, clipped to [-1, 1];
    ``pedal > 0`` is throttle, ``pedal < 0`` is brake.

    Update rule (``v`` = start-of-step speed; all forces are evaluated at ``v``):

    1. Steering: ``delta`` moves towards ``steer_cmd * max_steer_rad`` by at most
       ``steer_rate_rad_s * dt``.
    2. Grip ``a_max = grip_limit(v)``. Demanded path curvature ``k = tan(delta) / wheelbase``
       and lateral demand ``a_lat_dem = v^2 * k`` (kinematic bicycle).
    3. Tyre longitudinal request: throttle
       ``pedal * min(power / (mass * max(v, V_FLOOR)), max_drive_g * g)``; brake
       ``pedal * min(max_brake_g * g, a_max)`` (negative).
    4. Friction circle, lateral first: the tyre request is trimmed to
       ``sqrt(max(a_max^2 - a_lat_dem^2, 0))``. ``sliding`` is True when
       ``hypot(a_lat_dem, request) > a_max``. A longitudinal-only overrun (e.g. full throttle
       mid-corner) just loses the excess, like wheelspin or lock-up.
    5. Lateral overrun (``|a_lat_dem| > a_max``): no longitudinal tyre force is left, and the
       slide scrubs speed with an extra deceleration
       ``a_scrub = a_max * (1 - a_max / |a_lat_dem|)``, which is 0 at the limit and tends to
       ``a_max`` for a hopeless overspeed.
    6. Speed: ``a = tyre - drag_coeff * v^2 - a_scrub``; ``v_new = max(v + a * dt, 0)``
       (no reverse; braking at rest stays at 0).
    7. Yaw (semi-implicit, uses ``v_new``): ``yaw_rate = v_new * k``, with ``k`` clamped so
       ``|v_new * yaw_rate| <= grip_limit(v_new)``. That clamp also sets ``sliding``. When the
       car slides, it therefore runs wide of the steered radius.
    8. Position (semi-implicit, uses ``v_new`` and the updated yaw):
       ``x += v_new * cos(yaw_new) * dt``, ``y += v_new * sin(yaw_new) * dt``.

    The reported ``a_long`` and ``a_lat`` are the realised values of steps 6 and 7, so
    ``|a_lat| <= grip_limit(v_new)`` always holds. The friction circle limits tyre forces only;
    drag and scrub are added on top, so ``hypot(a_long, a_lat)`` can exceed ``a_max``.
    """
    if not dt > 0.0:
        raise ValueError(f"dt must be > 0, got {dt!r}")
    n = state.n
    steer = np.clip(np.broadcast_to(np.asarray(steer_cmd, dtype=np.float64), (n,)), -1.0, 1.0)
    pedal = np.clip(np.broadcast_to(np.asarray(pedal_cmd, dtype=np.float64), (n,)), -1.0, 1.0)
    v = state.v
    wheelbase = params.wheelbase_m

    # 1. Rate-limited steering.
    max_d_delta = params.steer_rate_rad_s * dt
    delta = state.delta + np.clip(
        steer * params.max_steer_rad - state.delta, -max_d_delta, max_d_delta
    )

    # 2. Grip and lateral demand at the start-of-step speed.
    a_max = grip_limit(v, params)
    k_dem = np.tan(delta) / wheelbase
    a_lat_dem = v * v * k_dem
    abs_lat_dem = np.abs(a_lat_dem)

    # 3. Tyre longitudinal request.
    a_drive = np.minimum(
        params.power_w / (params.mass_kg * np.maximum(v, V_FLOOR)), params.max_drive_g * G
    )
    a_brake = np.minimum(params.max_brake_g * G, a_max)
    request = np.where(pedal > 0.0, pedal * a_drive, pedal * a_brake)

    # 4. Friction circle: lateral has priority, longitudinal gets what is left.
    budget = np.sqrt(np.maximum(a_max * a_max - a_lat_dem * a_lat_dem, 0.0))
    tyre_long = np.clip(request, -budget, budget)
    sliding = np.hypot(a_lat_dem, request) > a_max

    # 5. Scrub when the lateral demand alone exceeds the limit (0 otherwise).
    a_scrub = a_max * (1.0 - a_max / np.maximum(abs_lat_dem, a_max))

    # 6. Speed, no reverse.
    a_long_cmd = tyre_long - params.drag_coeff * v * v - a_scrub
    v_new = np.maximum(v + a_long_cmd * dt, 0.0)

    # 7. Yaw with the updated speed; curvature clamped to the grip available at v_new.
    a_max_new = grip_limit(v_new, params)
    a_lat_cand = v_new * v_new * k_dem
    abs_lat_cand = np.abs(a_lat_cand)
    sliding |= abs_lat_cand > a_max_new
    k = k_dem * (a_max_new / np.maximum(abs_lat_cand, a_max_new))
    yaw_rate = v_new * k
    a_lat = v_new * yaw_rate
    a_long = (v_new - v) / dt

    # 8. Semi-implicit position update; write into the existing arrays (in place).
    yaw_new = state.yaw + yaw_rate * dt
    yaw_new = (yaw_new + np.pi) % (2.0 * np.pi) - np.pi
    state.x[...] = state.x + v_new * np.cos(yaw_new) * dt
    state.y[...] = state.y + v_new * np.sin(yaw_new) * dt
    state.yaw[...] = yaw_new
    state.v[...] = v_new
    state.delta[...] = delta

    return StepInfo(a_long=a_long, a_lat=a_lat, sliding=sliding)

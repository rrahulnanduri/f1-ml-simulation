"""Training view (M4): watch a run's generations drive while it trains.

Training stays a headless process (``f1sim train``): it runs at about 64x real time, and a
window drawing every evaluated car would hold it to 1x. After every generation the trainer
replaces ``generation.npz`` in the run folder (:meth:`~f1sim.train.metrics.RunWriter.
write_generation`): the genomes, their fitnesses and where each one's episode started. This
screen loads the newest one and drives it again here, one car per genome, in a
:class:`~f1sim.sim.world.World` built like the trainer's. The physics and the policy are
deterministic, so in stage 2 the cars on screen drive exactly the episodes the trainer scored
(in stage 1 each genome's first episode; the trainer drives ``episodes_per_genome`` from
random starts). When the episode is over (every car out or finished, or
:data:`LEADER_GRACE_S` after the first car finishes its lap) the screen moves on to the newest
generation, so it shows a generation every lap or so while many more train in between.

The testable logic is pure Python/numpy, without arcade objects:

- :class:`RunSetup` reads what a replay needs from the run's ``config.toml``.
- :func:`read_metrics` reads ``metrics.csv`` (tolerating a half-written last row).
- :class:`GenerationPlayback` drives one generation: fixed decision steps fed by real time
  times the playback speed, poses interpolated between the last two steps, each car's state
  and the live ranking.
- :class:`TrainingProcess` starts ``f1sim train`` in the background (no console; output in
  ``train.log``) and stops it.
- :class:`TrainingWatch` ties a run folder to the playback: it loads a new generation when
  one is due and says what the run is doing.

**Window.** The track is the cached :class:`~f1sim.app.render_track.TrackRenderer` under an
``arcade.Camera2D`` synced from :class:`~f1sim.app.camera.Camera`, as in the replay viewer.
It opens fitted to the whole circuit. All cars are pixel-art sprites of one size at every
zoom (:data:`CAR_LENGTH_PX`), each in its own colour (``theme.field_liveries``, also its
sliver in the tower), faded once out; the generation's best genome (by the trainer's
fitness) has a light outline and the only label, ``BEST``. The HUD
(:mod:`f1sim.app.ui.train_hud`) has the live order, the generation, the cars still running
with the episode clock, and the best lap per generation.

Controls: mouse wheel zoom at the cursor, right-drag pan, ``Z`` fit the whole track, ``F``
follow the generation's best car, Space pause, Up/Down playback speed, ``N`` jump to the
newest generation, ``Esc`` back to the menu (stopping the training this screen started).
"""

from __future__ import annotations

import atexit
import csv
import math
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import arcade
import numpy as np

from f1sim.app import shell
from f1sim.app.camera import Camera
from f1sim.app.drive import format_lap_time
from f1sim.app.shell import Screen
from f1sim.app.ui import theme
from f1sim.app.ui import timing as tm
from f1sim.sim.car_model import CarParams
from f1sim.sim.progress import DoneReason
from f1sim.track.track import Track
from f1sim.train.metrics import GENERATION_FILE, read_generation

MAX_FRAME_S = 0.25
"""Longest frame time fed to the playback (a stall doesn't jump the cars)."""
MAX_STEPS_PER_FRAME = 64
"""At most this many decision steps per frame; a slow machine plays slower, not jerkier."""
SPEED_STEPS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
"""Playback speeds selectable with Up/Down."""
DEFAULT_SPEED = 2.0
LEADER_GRACE_S = 3.0
"""Simulated seconds the episode carries on after the first car finishes its lap."""
POLL_S = 0.5
"""How often the run folder is checked for new files, s of real time."""
ZOOM_STEP = 1.15
"""Zoom factor per mouse-wheel notch."""
FOLLOW_PIXELS_PER_METRE = 4.0
"""Zoom the follow camera starts at when it's switched on from far out."""
FIT_PADDING_PX = 60
OUT_ALPHA = 70
"""Opacity (0-255) of cars that are out of the episode."""
TOWER_ROWS = 10
CAR_LENGTH_PX = 40.0
"""On-screen length of every car, at any zoom (one physical pixel per art pixel at 1x)."""


# ------------------------------------------------------------------------------- the run


@dataclass(frozen=True)
class RunSetup:
    """What re-driving a run's generations needs, from its ``config.toml``."""

    track_id: str
    hidden: tuple[int, ...]
    use_track_limits: bool
    car: CarParams
    dt: float
    action_repeat: int
    init: str = "clone"

    @classmethod
    def from_config(cls, path: Path) -> RunSetup:
        """Raises ``OSError`` or ``ValueError`` if the file is missing or incomplete."""
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        resolved = data.get("resolved", {})
        known = {f.name for f in fields(CarParams)}
        car = CarParams(**{k: v for k, v in data.get("car", {}).items() if k in known})
        try:
            return cls(
                track_id=str(data["track_id"]),
                hidden=tuple(int(h) for h in data["hidden"]),
                use_track_limits=bool(data.get("use_track_limits", True)),
                car=car,
                dt=float(resolved["dt"]),
                action_repeat=int(resolved["action_repeat"]),
                init=str(data.get("init", "clone")),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: incomplete run config ({exc})") from exc


def read_metrics(path: Path) -> list[dict[str, float]]:
    """``metrics.csv`` rows as floats (an empty field is nan). A row that doesn't parse (the
    trainer may be half-way through writing it) ends the list."""
    rows: list[dict[str, float]] = []
    try:
        with Path(path).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                try:
                    rows.append(
                        {k: (float(v) if v else math.nan) for k, v in raw.items() if k is not None}
                    )
                except (TypeError, ValueError):
                    break
    except OSError:
        return []
    return rows


def best_lap_history(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """``(gen, best_lap_s)`` of the generations that completed a lap."""
    gen = np.array([r.get("gen", math.nan) for r in rows], dtype=np.float64)
    lap = np.array([r.get("best_lap_s", math.nan) for r in rows], dtype=np.float64)
    ok = np.isfinite(gen) & np.isfinite(lap)
    return gen[ok], lap[ok]


def run_best_lap(rows: list[dict[str, float]]) -> float:
    """The run's fastest lap so far (nan before the first)."""
    _, lap = best_lap_history(rows)
    return float(lap.min()) if lap.size else math.nan


# ---------------------------------------------------------------------- the playback

_CRASHES = (DoneReason.OFF_TRACK, DoneReason.WRONG_WAY, DoneReason.STALL)


@dataclass(frozen=True)
class CarRow:
    """One car of the live order: ``index`` into the generation, its state (``"lap"``
    finished, ``"run"`` still driving, ``"out"`` crashed or stalled, ``"time"`` out of
    time), lap time (s, nan unless finished) and progress (m)."""

    index: int
    state: str
    lap_time: float
    progress: float


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class GenerationPlayback:
    """One generation driven again: car ``i`` is genome ``i``, from its recorded start.

    ``world`` must have one car per genome and the run's ``dt``, ``action_repeat``, episode
    cap and car; ``act(params, obs)`` is the policy. Real time goes in through
    :meth:`update` (times :attr:`speed`), in whole decision steps.
    """

    def __init__(self, world, act, generation: dict, speed: float = DEFAULT_SPEED):
        self.world = world
        self.act = act
        self.gen = int(generation["gen"])
        self.stage = int(generation["stage"])
        self.params = np.asarray(generation["params"], dtype=np.float32)
        self.fitness = np.asarray(generation["fitness"], dtype=np.float64)
        self.episode_s = float(generation["episode_s"])
        n = self.params.shape[0]
        if world.n_cars != n:
            raise ValueError(f"world has {world.n_cars} cars for {n} genomes")
        self.best = int(np.argmax(self.fitness))
        self.speed = float(speed)
        self.paused = False
        self.decision_dt = world.config.dt * world.config.action_repeat
        self._obs = world.reset(start_s=generation["start_s"], start_v=generation["start_v"])
        self.finished = np.zeros(n, dtype=bool)
        self.lap_time = np.full(n, np.nan)
        self.lap_progress = np.full(n, np.nan)
        self.first_finish_t: float | None = None
        self._acc = 0.0
        self._prev = self._pose()
        self.steps = 0

    # ------------------------------------------------------------------- state

    @property
    def n_cars(self) -> int:
        return self.params.shape[0]

    @property
    def t(self) -> float:
        """Simulated time since the start, s."""
        return self.steps * self.decision_dt

    @property
    def done(self) -> np.ndarray:
        return self.world.progress.state.done

    @property
    def running(self) -> np.ndarray:
        """Cars still driving their episode (not out, not finished)."""
        return ~self.done & ~self.finished

    @property
    def over(self) -> bool:
        """Every car out or finished, or :data:`LEADER_GRACE_S` after the first finish."""
        if not self.running.any():
            return True
        return self.first_finish_t is not None and self.t >= self.first_finish_t + LEADER_GRACE_S

    @property
    def alpha(self) -> float:
        """How far the drawn moment is between the last two steps (0..1)."""
        return min(max(self._acc / self.decision_dt, 0.0), 1.0)

    def _pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        car = self.world.car
        return car.x.copy(), car.y.copy(), car.yaw.copy()

    def poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(x, y, yaw)`` arrays to draw, between the last two steps by :attr:`alpha`."""
        a = self.alpha
        x0, y0, yaw0 = self._prev
        car = self.world.car
        yaw = yaw0 + a * _wrap(car.yaw - yaw0)
        return x0 + a * (car.x - x0), y0 + a * (car.y - y0), yaw

    def progress(self) -> np.ndarray:
        """Metres along the track since the start (frozen at the line for finished cars)."""
        prog = self.world.progress.state.progress
        return np.where(self.finished, self.lap_progress, prog)

    def states(self) -> list[str]:
        reasons = self.world.progress.state.done_reason
        out = []
        for i in range(self.n_cars):
            if self.finished[i]:
                out.append("lap")
            elif not self.done[i]:
                out.append("run")
            elif reasons[i] in _CRASHES:
                out.append("out")
            else:
                out.append("time")
        return out

    def order(self) -> list[CarRow]:
        """The live order: finished cars by lap time, then the rest by progress."""
        prog = self.progress()
        states = self.states()
        rows = [
            CarRow(i, states[i], float(self.lap_time[i]), float(prog[i]))
            for i in range(self.n_cars)
        ]
        return sorted(
            rows,
            key=lambda r: (0, r.lap_time) if r.state == "lap" else (1, -r.progress),
        )

    # ------------------------------------------------------------------ update

    def step(self) -> None:
        """One decision step for every car."""
        self._prev = self._pose()
        actions = self.act(self.params, self._obs)
        res = self.world.step(actions)
        self._obs = res.obs
        self.steps += 1
        newly = res.lap_completed & ~self.finished
        if newly.any():
            self.lap_time[newly] = res.lap_time[newly]
            self.lap_progress[newly] = self.world.progress.state.progress[newly]
            self.finished |= newly
            if self.first_finish_t is None:
                self.first_finish_t = self.t

    def update(self, dt: float) -> None:
        """Advance by ``dt`` seconds of real time (unless paused, or over)."""
        if self.paused:
            return
        self._acc += min(max(dt, 0.0), MAX_FRAME_S) * self.speed
        n = 0
        while self._acc >= self.decision_dt and not self.over and n < MAX_STEPS_PER_FRAME:
            self.step()
            self._acc -= self.decision_dt
            n += 1
        if self.over or n == MAX_STEPS_PER_FRAME:
            self._acc = min(self._acc, self.decision_dt)


def faster(speed: float) -> float:
    return next((s for s in SPEED_STEPS if s > speed + 1e-9), SPEED_STEPS[-1])


def slower(speed: float) -> float:
    return next((s for s in reversed(SPEED_STEPS) if s < speed - 1e-9), SPEED_STEPS[0])


# ---------------------------------------------------------------------- the process


def new_run_dir(track_id: str, runs_root: Path) -> Path:
    """``runs/<track>-<timestamp>``, as ``f1sim train`` names it (``-2``, ``-3``... if taken)."""
    base = Path(runs_root) / f"{track_id}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir, k = base, 2
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}-{k}")
        k += 1
    return run_dir


def train_command(track_id: str, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "f1sim.cli",
        "train",
        "--track",
        track_id,
        "--run-dir",
        str(run_dir),
    ]


class TrainingProcess:
    """``f1sim train`` in the background: no console window, output in ``train.log`` in the
    run folder. It's stopped by :meth:`stop` and when the app exits."""

    def __init__(self, track_id: str, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train.log"
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with self.log_path.open("wb") as log:
            self.proc = subprocess.Popen(
                train_command(track_id, self.run_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=flags,
            )
        self.stopped = False
        atexit.register(self.stop)

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.proc.poll()

    def stop(self) -> None:
        if self.running:
            self.stopped = True
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def status_text(
    process: TrainingProcess | None, has_generation: bool, setup: RunSetup | None
) -> tuple[str, bool]:
    """``(status, is_problem)`` of the run for the HUD."""
    if process is None:
        return ("Watching" if has_generation else "Waiting for a generation"), False
    code = process.returncode
    if code is None:
        if has_generation:
            return "Training", False
        if setup is None:
            return "Copying the scripted driver", False
        return "Evaluating the first generation", False
    if process.stopped:
        return "Stopped", False
    if code == 0:
        return "Finished", False
    return f"Training failed: see {process.log_path.name}", True


# ------------------------------------------------------------------------ the watch


class TrainingWatch:
    """A run folder, its newest generation and the playback of the one on screen.

    ``make_world(n_cars, episode_s)`` builds the World for a generation (cached by its
    arguments) and ``act(spec_hidden)`` returns the policy for a run's hidden sizes; both are
    injected so this class needs no window, and tests can drive it with a small track.
    """

    def __init__(self, run_dir: Path, make_world, make_act, process: TrainingProcess | None = None):
        self.run_dir = Path(run_dir)
        self.make_world = make_world
        self.make_act = make_act
        self.process = process
        self.setup: RunSetup | None = None
        self.playback: GenerationPlayback | None = None
        self.metrics: list[dict[str, float]] = []
        self.newest_gen: int | None = None
        self.speed = DEFAULT_SPEED
        self._worlds: dict[tuple[int, float], object] = {}
        self._act = None
        self._mtimes: dict[str, float] = {}
        self._poll_t = math.inf

    def _changed(self, name: str) -> bool:
        try:
            m = (self.run_dir / name).stat().st_mtime
        except OSError:
            return False
        if self._mtimes.get(name) == m:
            return False
        self._mtimes[name] = m
        return True

    def poll(self) -> bool:
        """Re-read the folder's files that changed; True if a newer generation is there."""
        if self.setup is None and self._changed("config.toml"):
            try:
                self.setup = RunSetup.from_config(self.run_dir / "config.toml")
            except (OSError, ValueError):
                self._mtimes.pop("config.toml", None)  # try again next poll
        if self._changed("metrics.csv"):
            self.metrics = read_metrics(self.run_dir / "metrics.csv")
            if self.metrics:
                self.newest_gen = int(self.metrics[-1]["gen"])
        on_screen = None if self.playback is None else self.playback.gen
        return self.newest_gen is not None and self.newest_gen != on_screen

    def load_newest(self) -> bool:
        """Start playing ``generation.npz``; False if there's none (or no config yet)."""
        if self.setup is None:
            return False
        generation = read_generation(self.run_dir / GENERATION_FILE)
        if generation is None:
            return False
        n = int(np.asarray(generation["params"]).shape[0])
        key = (n, round(float(generation["episode_s"]), 6))
        world = self._worlds.get(key)
        if world is None:
            world = self._worlds[key] = self.make_world(self.setup, *key)
        if self._act is None:
            self._act = self.make_act(self.setup)
        self.playback = GenerationPlayback(world, self._act, generation, self.speed)
        return True

    def update(self, dt: float) -> bool:
        """Advance the playback and, every :data:`POLL_S`, the folder; True when a new
        generation was loaded (the view rebuilds what depends on it)."""
        self._poll_t += dt
        newer = False
        if self._poll_t >= POLL_S:
            self._poll_t = 0.0
            newer = self.poll()
        loaded = False
        pb = self.playback
        if pb is None:
            loaded = newer and self.load_newest()
        elif pb.over and not pb.paused:
            loaded = self.load_newest()  # the newest generation, or this one again
        if self.playback is not None:
            self.playback.update(dt)
        return loaded

    def skip_to_newest(self) -> bool:
        self.poll()
        return self.load_newest()

    def set_speed(self, speed: float) -> None:
        self.speed = speed
        if self.playback is not None:
            self.playback.speed = speed

    def status(self) -> tuple[str, bool]:
        return status_text(self.process, self.newest_gen is not None, self.setup)


def default_make_world(track: Track):
    """The World factory the window uses: the trainer's World for ``track``."""
    from f1sim.sim.world import World, WorldConfig

    def make(setup: RunSetup, n_cars: int, episode_s: float):
        sim_track = track.sim_track() if setup.use_track_limits else track
        return World(
            sim_track,
            WorldConfig(
                n_cars=n_cars,
                dt=setup.dt,
                action_repeat=setup.action_repeat,
                episode_time_s=episode_s,
                car=setup.car,
                use_track_limits=False,
            ),
        )

    return make


def default_make_act(setup: RunSetup):
    from f1sim.agents import policy

    spec = policy.PolicySpec(hidden=setup.hidden)
    return lambda params, obs: policy.act(spec, params, obs)


# ------------------------------------------------------------------------ the HUD data


def tower_rows(order: list[CarRow], best: int, colours: list[tuple]) -> list[tm.TowerRow]:
    """The first :data:`TOWER_ROWS` of the live order: the leader's lap time or distance,
    the others' gap to it (``+0.123`` behind a lap time), their distance, ``OUT`` or
    ``TIME``. Each row's sliver is its car's colour (``colours[index]``)."""
    rows = []
    leader_lap = order[0].lap_time if order and order[0].state == "lap" else None
    for k, r in enumerate(order[:TOWER_ROWS]):
        if r.state == "lap":
            value = (
                format_lap_time(r.lap_time)
                if leader_lap is None or k == 0
                else tm.format_gap(r.lap_time - leader_lap)
            )
        elif r.state == "out":
            value = "OUT"
        elif r.state == "time":
            value = "TIME"
        else:
            value = f"{max(r.progress, 0.0):,.0f} m"
        name = "Best" if r.index == best else f"Car {r.index + 1}"
        rows.append(tm.TowerRow(str(k + 1), name, value, colours[r.index], r.index == best))
    return rows


# --------------------------------------------------------------------------- window


def controls(on_back) -> list[str]:
    return [*CONTROLS[:-1], "Esc back" if on_back is not None else "Esc quit"]


CONTROLS = ["Wheel zoom", "Z fit track", "F follow best", "↑↓ speed", "N newest", "Esc quit"]


class TrainingView(Screen):
    """The training screen: a :class:`TrainingWatch` on a track, with the HUD. Esc goes back
    (``on_back``), stopping the training process this screen was given."""

    def __init__(
        self,
        track: Track,
        run_dir: Path,
        process: TrainingProcess | None = None,
        glass: bool = True,
        on_back=None,
    ):
        super().__init__(on_back)
        self.title = f"f1sim training - {track.data.name}"
        from f1sim.app.render_track import TrackRenderer
        from f1sim.app.ui.glass import GlassRenderer
        from f1sim.app.ui.train_hud import TrainingHud

        self.background_color = theme.BACKGROUND
        self.track = track
        self.watch = TrainingWatch(
            run_dir, default_make_world(track), default_make_act, process=process
        )
        self.camera = Camera(self.width, self.height)
        self.follow = False
        self.camera2d = arcade.camera.Camera2D()
        self.track_view = TrackRenderer(track, self.ctx)

        theme.load_fonts()
        self.glass = GlassRenderer(self.window, enabled=glass)
        self.hud = TrainingHud(self.glass, TOWER_ROWS, controls(on_back))
        self.track_title = track.data.id.replace("_", " ").upper()
        self.hud.headline.set([self.track_title, "Training"])
        self.cars = None
        self._painted_best: int | None = None
        self._car_colours: list[tuple] = []
        self._metrics_len = -1
        self.hud.layout(self.width, self.height)
        self.fit_track()

    # ------------------------------------------------------------------------ helpers

    def fit_track(self) -> None:
        """The whole circuit in the area the HUD leaves free; the follow camera goes off."""
        self.follow = False
        walls = np.vstack([self.track.left_wall(), self.track.right_wall()])
        self.camera.fit(*walls.min(axis=0), *walls.max(axis=0), padding_px=FIT_PADDING_PX)
        cx, cy = self.camera.centre
        self.camera.follow_at(cx, cy, *self.hud.stage_centre())

    def _toggle_follow(self) -> None:
        self.follow = not self.follow
        if self.follow and self.camera.pixels_per_metre < FOLLOW_PIXELS_PER_METRE:
            self.camera.zoom_at(
                FOLLOW_PIXELS_PER_METRE / self.camera.pixels_per_metre,
                0.5 * self.camera.viewport_w,
                0.5 * self.camera.viewport_h,
            )
        self._follow_camera()

    def _follow_camera(self) -> None:
        pb = self.watch.playback
        if self.follow and pb is not None:
            x, y, _ = pb.poses()
            self.camera.follow_at(float(x[pb.best]), float(y[pb.best]), *self.hud.stage_centre())

    def _build_cars(self) -> None:
        """Sprites for the generation on screen (rebuilt when the population size changes)."""
        from f1sim.app.ui.pixelcar import CarLayer

        pb = self.watch.playback
        n = pb.n_cars
        if self.cars is None or len(self.cars.labels) != n:
            # A colour per car; only the highlighted (best) car's label is drawn, so every
            # label says so
            self._liveries = theme.field_liveries(n)
            self._car_colours = [lv.primary for lv in self._liveries]
            self.cars = CarLayer(
                ["Best"] * n, self._liveries, [False] * n, fixed_length_px=CAR_LENGTH_PX
            )
            self.cars.only_highlight_label = True
            self._painted_best = None
        if self._painted_best != pb.best:
            from f1sim.app.ui.pixelcar import car_texture

            # The best car keeps its colour and gets the light outline
            sprites = self.cars.sprites.sprites
            if self._painted_best is not None:
                old = self._painted_best
                sprites[old].texture = car_texture(self._liveries[old], False)
            sprites[pb.best].texture = car_texture(self._liveries[pb.best], True)
            self._painted_best = pb.best

    # ------------------------------------------------------------------------- events

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        pb = self.watch.playback
        if symbol == arcade.key.ESCAPE:
            self.close()
        elif symbol == arcade.key.SPACE and pb is not None:
            pb.paused = not pb.paused
        elif symbol == arcade.key.UP:
            self.watch.set_speed(faster(self.watch.speed))
        elif symbol == arcade.key.DOWN:
            self.watch.set_speed(slower(self.watch.speed))
        elif symbol in (arcade.key.Z, arcade.key.HOME):
            self.fit_track()
        elif symbol == arcade.key.F:
            self._toggle_follow()
        elif symbol == arcade.key.N:
            if self.watch.skip_to_newest():
                self._build_cars()

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        if scroll_y:
            self.camera.zoom_at(ZOOM_STEP**scroll_y, x, y)  # follow re-centres on the car

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        if buttons & arcade.MOUSE_BUTTON_RIGHT:
            self.follow = False
            self.camera.pan(dx, dy)

    def on_resize(self, width: int, height: int) -> None:
        self.camera.resize(width, height)
        self.camera2d.match_window()
        self.hud.layout(width, height)
        self._follow_camera()

    def on_hide_view(self) -> None:
        if self.watch.process is not None:
            self.watch.process.stop()

    def on_update(self, delta_time: float) -> None:
        if self.watch.update(delta_time):
            self._build_cars()
        self._follow_camera()

    # --------------------------------------------------------------------------- draw

    def on_draw(self) -> None:
        now = time.perf_counter()
        pb = self.watch.playback
        self._update_hud(now)
        with self.glass.scene():
            self.camera2d.position = self.camera.centre
            self.camera2d.zoom = self.camera.pixels_per_metre
            self.glass.use_camera(self.camera2d)
            self.track_view.draw(self.camera.pixels_per_metre)
            self.glass.use_pixels()
            self.track_view.draw_ticks(self.camera, self.glass)
            if pb is not None and self.cars is not None:
                x, y, yaw = pb.poses()
                running = pb.running | pb.finished
                for i, sp in enumerate(self.cars.sprites.sprites):
                    alpha = 255 if running[i] else OUT_ALPHA
                    if sp.alpha != alpha:
                        sp.alpha = alpha
                self.cars.draw(
                    self.camera,
                    list(zip(x.tolist(), y.tolist(), yaw.tolist(), strict=True)),
                    pb.best,
                    self.get_pixel_ratio(),
                )
        self.glass.present()
        self.hud.draw(now)

    def _update_hud(self, now: float) -> None:
        from f1sim.app.ui.hud import ticker_date_time

        watch = self.watch
        pb = watch.playback
        status, problem = watch.status()
        mode = "PAUSED" if pb is not None and pb.paused else f"{watch.speed:g}×"
        self.hud.ticker.set([*ticker_date_time(), self.track_title, status.upper()], now)
        if len(watch.metrics) != self._metrics_len:
            self._metrics_len = len(watch.metrics)
            gens, laps = best_lap_history(watch.metrics)
            self.hud.progress.set(gens, laps, run_best_lap(watch.metrics))
        if pb is None:
            self.hud.tower.set("Generation", "", [])
            self.hud.generation.set(None, None, "", watch.newest_gen)
            self.hud.field.set(0, 0, "", mode)
        else:
            order = pb.order()
            self.hud.tower.set(
                f"Generation {pb.gen}",
                "Live order",
                tower_rows(order, pb.best, self._car_colours),
            )
            lap = pb.lap_time[pb.best]
            best_txt = (
                f"best {format_lap_time(lap)}"
                if math.isfinite(lap)
                else f"best {max(order[0].progress, 0.0):,.0f} m"
            )
            self.hud.generation.set(pb.gen, pb.stage, best_txt, watch.newest_gen)
            n_on = int(pb.running.sum())
            self.hud.field.set(n_on, pb.n_cars, f"{pb.t:4.1f} s", mode)
        colour = theme.ACCENT if problem else theme.TEXT_PRIMARY
        if self.follow:
            self.hud.footer.set_status(f"{status} · following best", colour)
        else:
            self.hud.footer.set_status(status, colour)


# ----------------------------------------------------------------------------- entry


def run(track_id: str, run_dir: Path) -> None:
    """Open the training view on an existing run folder (``f1sim watch``); prints a message
    and returns if the track isn't built."""
    from f1sim.data.track_pipeline import load_track

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        print(f"Run folder not found: {run_dir}")
        return
    try:
        track = load_track(track_id)
    except FileNotFoundError:
        print(
            f"Track '{track_id}' hasn't been built yet. Build it first with:\n"
            f"    f1sim build-track {track_id}"
        )
        return
    shell.run(lambda window: TrainingView(track, run_dir))

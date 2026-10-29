"""What the training view's "How it works" drawer says (pure Python, no window).

The drawer (:class:`~f1sim.app.ui.drawer.Drawer`) explains the design decisions behind the
training in plain words, on four pages: how the cars learn, what they are scored on, how the
run compares with the real fastest lap, and how real the physics is. :func:`pages` writes
them from :class:`Facts`, the live state of the run, so the text uses this run's numbers
(the generation on screen, its best car's score worked out, this car's power and grip)
rather than generic ones. The sources are decisions D8, D11, D16 and D17 in
docs/PROJECT_LOG.md.

A page is a list of blocks: :class:`Heading`, :class:`Para`, :class:`Row` (a name and a
value) and :class:`Bar` (a labelled lap time against a scale). :func:`opening_page` picks the
page that fits what the run is doing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from f1sim.sim.car_model import CarParams
from f1sim.train.trainer import CRASH_PENALTY, LAP_BONUS, LAP_TIME_WEIGHT, TrainConfig

REFERENCE_POLES: dict[str, tuple[float, str]] = {
    "silverstone": (85.819, "Russell, 2024 pole"),
}
"""Real fastest laps used when the track has no real-driver ghost (only ones this project
has checked)."""


@dataclass(frozen=True)
class Heading:
    text: str


@dataclass(frozen=True)
class Para:
    text: str
    live: bool = False  # drawn brighter: it describes this run, now


@dataclass(frozen=True)
class Row:
    name: str
    value: str


@dataclass(frozen=True)
class Bar:
    label: str
    lap_s: float
    highlight: bool = False


@dataclass(frozen=True)
class Page:
    title: str
    blocks: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class Facts:
    """The live state of the run the drawer talks about; ``None`` / nan where not known."""

    status: str = ""
    copying: bool = False  # still copying the scripted driver, no generation yet
    gen: int | None = None
    stage: int | None = None
    population: int | None = None
    sigma: float = math.nan
    best_index: int | None = None
    best_lap_s: float = math.nan  # the best car of the generation on screen
    best_progress_m: float = math.nan
    best_crashed: bool = False
    best_fitness: float = math.nan  # the trainer's score, steering cost included
    episode_s: float = math.nan
    track_length_m: float = math.nan
    run_best_s: float = math.nan
    t_ref_s: float = math.nan
    pole_s: float = math.nan
    pole_label: str = ""
    car: CarParams = field(default_factory=CarParams)


def fmt_lap(t: float) -> str:
    if not math.isfinite(t):
        return "-:--.---"
    m, s = divmod(t, 60.0)
    return f"{int(m)}:{s:06.3f}"


def pct(a: float, b: float) -> str:
    """``a`` against ``b`` as ``+0.7%`` / ``-0.8%``."""
    return f"{(a / b - 1.0) * 100.0:+.1f}%"


def lap_score(track_length_m: float, episode_s: float, lap_s: float) -> float:
    """The trainer's score for a completed lap (trainer module docstring, stage 2)."""
    return track_length_m + LAP_BONUS + LAP_TIME_WEIGHT * (episode_s - lap_s)


def top_speed_mps(car: CarParams) -> float:
    """Where power-limited drive equals drag: ``P / (m v) = c v^2``."""
    if car.drag_coeff <= 0 or car.mass_kg <= 0:
        return math.inf
    return (car.power_w / (car.mass_kg * car.drag_coeff)) ** (1.0 / 3.0)


def grip_g(car: CarParams, v_mps: float) -> float:
    return car.grip_base_g + car.grip_aero_coeff * v_mps * v_mps


# ----------------------------------------------------------------------------- pages


def _learn(f: Facts) -> Page:
    blocks: list = []
    if f.copying:
        blocks.append(Para("Now: copying the scripted driver. This takes about a minute.", True))
    elif f.gen is not None:
        n = f.population or 0
        step = f" Size of the random changes: {f.sigma:.4f}." if math.isfinite(f.sigma) else ""
        blocks.append(Para(f"Now: generation {f.gen}, {n} cars, each one lap.{step}", True))
    blocks += [
        Heading("1 · Copy a driver"),
        Para(
            "A scripted driver follows a racing line at a set speed. "
            "A small AI watches it and learns to copy its steering and pedals. "
            "Now the car can already drive a lap."
        ),
        Heading("2 · Evolve the copy"),
        Para(
            "Each generation is a group of cars that are all a little different. They all "
            "drive one lap from the same start. The best few are kept, and the next group "
            "is made from them with small random changes. The changes get bigger when many "
            "of them help, and smaller when few do."
        ),
        Heading("What a car sees"),
        Para(
            "29 numbers, 30 times a second: how far the track edges are (9 beams), its "
            "speed, its angle and place on the track, the bends ahead, what it just did, "
            "and where the racing line goes next and how fast to go. It answers with "
            "steering and pedals."
        ),
        Heading("Why copy first"),
        Para(
            "Learning from nothing only reached 1:40.7 at Silverstone. The copy laps in "
            "about 1:27 straight away, so the cars spend their time getting faster, not "
            "learning to stay on the road. The catch: the car is told the racing line. It "
            "does not find it by itself."
        ),
    ]
    return Page("How they learn", tuple(blocks))


def _score(f: Facts) -> Page:
    blocks: list = [
        Para("Every car gets one score for its lap. A higher score is better."),
        Row("Lap completed", f"length + {LAP_BONUS:,.0f} + {LAP_TIME_WEIGHT:g} per s saved"),
        Row("No lap", "metres driven"),
        Row("Crash", f"metres driven − {CRASH_PENALTY:.0%}"),
        Row("Steering", f"− {TrainConfig.steer_change_weight:g} × each change², summed"),
        Para(
            "A car that finishes the lap always scores more than one that does not. After "
            f"that, every second saved is worth {LAP_TIME_WEIGHT:g} points."
        ),
    ]
    if f.best_index is not None and math.isfinite(f.track_length_m):
        car = f"Car {f.best_index + 1}"
        if math.isfinite(f.best_lap_s) and math.isfinite(f.episode_s):
            score = lap_score(f.track_length_m, f.episode_s, f.best_lap_s)
            saved = f.episode_s - f.best_lap_s
            steering = score - f.best_fitness if math.isfinite(f.best_fitness) else 0.0
            tail = (
                f", less {steering:,.0f} for steering = {f.best_fitness:,.0f}"
                if steering >= 0.5
                else ""
            )
            blocks.append(
                Para(
                    f"This generation's best, {car}: {fmt_lap(f.best_lap_s)}, "
                    f"{saved:.1f} s under the {f.episode_s:.0f} s limit = "
                    f"{f.track_length_m:,.0f} + {LAP_BONUS:,.0f} + "
                    f"{LAP_TIME_WEIGHT:g} × {saved:.1f} = {score:,.0f} points{tail}.",
                    True,
                )
            )
        elif math.isfinite(f.best_progress_m):
            crash = ", less 10% for crashing" if f.best_crashed else ""
            blocks.append(
                Para(
                    f"This generation's best, {car}, drove {f.best_progress_m:,.0f} m{crash}.",
                    True,
                )
            )
    blocks += [
        Heading("Crashes"),
        Para(
            "Off track: outside the track edges, which come from real drivers' fast laps. "
            "Wrong way: going backwards for a second. Stuck: hardly moving for a while."
        ),
        Heading("Smooth steering"),
        Para(
            "Every change to the steering costs a little, so a car that jerks the wheel "
            "loses about as much as a few tenths of a second. Without this the steering "
            "would wobble, because wobbling costs no time."
        ),
        Heading("Not scored"),
        Para("Tyre wear, fuel and touching other cars: the cars drive through each other."),
    ]
    return Page("What earns points", tuple(blocks))


def _results(f: Facts) -> Page:
    bars = []
    if math.isfinite(f.run_best_s):
        bars.append(Bar("This run", f.run_best_s, True))
    if math.isfinite(f.pole_s):
        bars.append(Bar(f"Real: {f.pole_label}" if f.pole_label else "Real lap", f.pole_s))
    if math.isfinite(f.t_ref_s):
        bars.append(Bar("Scripted driver", f.t_ref_s))
    blocks: list = []
    if math.isfinite(f.run_best_s) and math.isfinite(f.pole_s):
        gap = f.run_best_s - f.pole_s
        word = "faster than" if gap < 0 else "slower than"
        blocks.append(
            Para(
                f"Best lap so far is {abs(gap):.3f} s {word} the real lap "
                f"({pct(f.run_best_s, f.pole_s)}).",
                True,
            )
        )
    elif not math.isfinite(f.run_best_s):
        blocks.append(Para("No completed lap yet in this run.", True))
    if not math.isfinite(f.pole_s):
        blocks.append(
            Para("There is no real lap for this track yet. Make one with: f1sim ghost <track> 2024")
        )
    blocks += bars
    blocks += [
        Heading("How to read it"),
        Para(
            "The grip and braking were set so the scripted driver lands close to the real "
            "pole lap at Silverstone (+0.7%). So that match is partly on purpose. In 20 "
            "minutes the AI gets 0.3-0.8% under the real pole lap."
        ),
        Para(
            "So the AI is faster than a real driver here. Some of that is good driving. "
            "Some of it comes from the simple parts of the model: tyres that never spin, "
            "perfect steering and wide track edges. Treat it as a fast lap for this model, "
            "not proof that a real car could do it."
        ),
    ]
    return Page("How close to real", tuple(blocks))


def _physics(f: Facts) -> Page:
    car = f.car
    v_top = top_speed_mps(car)
    blocks: list = [
        Heading("Like a real car"),
        Row("Power / mass", f"{car.power_w / 1000:,.0f} kW / {car.mass_kg:,.0f} kg"),
        Row("Top speed", f"{v_top * 3.6:,.0f} km/h" if math.isfinite(v_top) else "-"),
        Row(
            "Grip at 80 / 290 km/h",
            f"{grip_g(car, 80 / 3.6):.1f} g / {grip_g(car, 290 / 3.6):.1f} g",
        ),
        Row("Braking limit", f"{car.max_brake_g:.1f} g"),
        Para(
            "Grip grows with speed, like downforce does. Turning and braking share the same "
            "grip: ask for too much and the car slides wide and slows down. The track "
            "width, edges and racing line are real."
        ),
        Heading("Simpler than a real car"),
        Para(
            "Tyres: they never slip sideways or shift weight, so the car cannot spin. "
            f"Braking is capped at {car.max_brake_g:.0f} g (real cars reach 5 g or more) so "
            "the stopping distances come out right. The track is flat: no hills, sloped "
            "corners or kerb bumps. No tyre wear, fuel, gears, hybrid power or moving wings."
        ),
        Para(
            "The AI also knows exactly where it is and never reacts late. The track edges "
            "sit about half a car width outside the white line."
        ),
        Heading("Most worth adding"),
        Para(
            "Tyres that can spin, track edges at the white line, a penalty for jerky "
            "steering, then real braking with matching grip."
        ),
    ]
    return Page("How real the physics is", tuple(blocks))


def pages(f: Facts) -> list[Page]:
    return [_learn(f), _score(f), _results(f), _physics(f)]


PAGE_TABS = ("Learn", "Score", "Results", "Physics")
"""Short names for the drawer's tabs, one per page of :func:`pages`."""


def opening_page(f: Facts) -> int:
    """The page to show when the drawer first opens: how they learn while the run is still
    copying, the results once it has a lap, else the score."""
    if f.copying or f.gen is None:
        return 0
    return 2 if math.isfinite(f.run_best_s) else 1

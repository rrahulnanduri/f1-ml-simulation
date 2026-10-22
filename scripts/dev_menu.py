"""Live reload for the main menu: relaunch ``f1sim menu`` whenever a source file changes.

    python scripts/dev_menu.py            (Ctrl+C to stop)

Watches ``src/f1sim`` (Python files and the bundled assets) and the hero renders in
``assets_src`` by polling modification times
(no extra dependency). When something changes it waits ``SETTLE_S`` for the save to finish,
closes the running menu and opens a new one. The window keeps its place and size across
restarts: the menu stores them in the file named by ``F1SIM_MENU_STATE``, which this script
sets.

If the menu exits with an error (a typo mid-edit, say), the traceback is printed and the next
change starts it again. Closing the window yourself (Esc or the close button) stops the
watcher too.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WATCH = ROOT / "src" / "f1sim"
RENDERS = ROOT / "assets_src"
"""The hero's renders (scripts/hero/render_top.py): a re-render reloads the menu too."""
SUFFIXES = {".py", ".ttf", ".png", ".glsl", ".json"}
POLL_S = 0.4
SETTLE_S = 0.3
STATE_FILE = Path(tempfile.gettempdir()) / "f1sim_menu_state.json"


def snapshot(roots=(WATCH, RENDERS)) -> dict[Path, float]:
    """Modification time of every watched file under ``roots``."""
    out = {}
    for p in (p for root in roots if root.is_dir() for p in root.rglob("*")):
        if p.suffix in SUFFIXES and "__pycache__" not in p.parts:
            try:
                out[p] = p.stat().st_mtime
            except OSError:
                pass
    return out


def changed(old: dict[Path, float], new: dict[Path, float]) -> list[Path]:
    """Files added, removed or modified between two snapshots."""
    return sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))


def start() -> subprocess.Popen:
    env = dict(os.environ, F1SIM_MENU_STATE=str(STATE_FILE))
    return subprocess.Popen([sys.executable, "-m", "f1sim.cli", "menu"], cwd=ROOT, env=env)


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    print(f"watching {WATCH.relative_to(ROOT)} - Ctrl+C to stop", flush=True)
    seen = snapshot()
    proc: subprocess.Popen | None = start()
    try:
        while True:
            time.sleep(POLL_S)
            if proc is not None and proc.poll() is not None:
                if proc.returncode == 0:
                    print("menu closed - stopping", flush=True)
                    return 0
                print(f"menu exited with code {proc.returncode} - waiting for a change", flush=True)
                proc = None
            now = snapshot()
            diff = changed(seen, now)
            if not diff:
                continue
            time.sleep(SETTLE_S)  # let the editor finish writing
            seen = snapshot()
            names = ", ".join(p.relative_to(ROOT).as_posix() for p in diff[:3])
            more = f" (+{len(diff) - 3})" if len(diff) > 3 else ""
            print(f"{time.strftime('%H:%M:%S')}  changed {names}{more} - reloading", flush=True)
            if proc is not None:
                stop(proc)
            proc = start()
    except KeyboardInterrupt:
        return 0
    finally:
        if proc is not None:
            stop(proc)


if __name__ == "__main__":
    raise SystemExit(main())

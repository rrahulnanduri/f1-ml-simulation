"""The app window's helpers (f1sim.app.shell) and the screens' footer labels: no window."""

from f1sim.app import drive, replay, shell


def test_window_state_round_trips_through_the_env_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    monkeypatch.setenv(shell.STATE_ENV, str(path))
    assert shell.load_state() == {}  # no file yet
    shell.save_state({"x": 10, "y": 20, "w": 1400, "h": 900})
    assert shell.load_state() == {"x": 10, "y": 20, "w": 1400, "h": 900}
    path.write_text("not json", encoding="utf-8")
    assert shell.load_state() == {}
    monkeypatch.delenv(shell.STATE_ENV)
    shell.save_state({"x": 1})  # no env: nothing saved, nothing raised
    assert shell.load_state() == {}


def test_esc_label_says_back_from_the_menu_and_quit_on_its_own():
    for module in (drive, replay):
        assert module.controls(on_back=lambda: None)[-1] == "Esc back"
        assert module.controls(on_back=None)[-1] == "Esc quit"
        assert module.controls(None)[:-1] == module.CONTROLS[:-1]


def test_only_restore_window_positions_on_a_screen():
    screens = [(0, 0, 2560, 1440), (2560, 0, 2560, 1440), (-2797, 1440, 1440, 900)]
    assert shell.on_a_screen(100, 100, 1280, screens)
    assert shell.on_a_screen(2600, 50, 1280, screens)  # second monitor
    assert shell.on_a_screen(-2700, 1500, 1280, screens)  # the one below-left
    assert not shell.on_a_screen(-32000, -32000, 1280, screens)  # parked (minimised)
    assert not shell.on_a_screen(6000, 100, 1280, screens)  # a monitor that's gone
    assert not shell.on_a_screen(100, 1420, 1280, screens)  # title bar off the bottom
    assert not shell.on_a_screen(100, 100, 1280, [])  # no screen info: don't restore

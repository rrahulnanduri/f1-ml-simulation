from pathlib import Path

import pytest

from f1sim import paths

SUBDIRS = {
    paths.tum_dir: "tum",
    paths.tracks_dir: "tracks",
    paths.ghosts_dir: "ghosts",
    paths.fastf1_dir: "fastf1",
}


def test_env_override_creates_dirs(tmp_path, monkeypatch):
    root = tmp_path / "cache root"
    monkeypatch.setenv(paths.ENV_VAR, str(root))
    assert paths.cache_dir() == root
    assert root.is_dir()
    for func, name in SUBDIRS.items():
        d = func()
        assert isinstance(d, Path)
        assert d == root / name
        assert d.is_dir()


def test_env_read_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "a"))
    assert paths.tracks_dir() == tmp_path / "a" / "tracks"
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "b"))
    assert paths.tracks_dir() == tmp_path / "b" / "tracks"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_default_uses_platformdirs(tmp_path, monkeypatch, value):
    calls = []

    def fake_user_cache_dir(appname, appauthor=None):
        calls.append((appname, appauthor))
        return str(tmp_path / "platform")

    if value is None:
        monkeypatch.delenv(paths.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(paths.ENV_VAR, value)
    monkeypatch.setattr(paths.platformdirs, "user_cache_dir", fake_user_cache_dir)
    assert paths.cache_dir() == tmp_path / "platform"
    assert paths.ghosts_dir() == tmp_path / "platform" / "ghosts"
    assert calls[0] == ("f1sim", False)


def test_existing_dir_is_fine(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert paths.fastf1_dir() == paths.fastf1_dir()


@pytest.mark.parametrize("bad", ["", "../evil", "a/b", "a\b", "C:x", "x.y", "a b"])
def test_track_file_refuses_ids_that_are_not_plain_names(tmp_path, monkeypatch, bad):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    with pytest.raises(ValueError, match="Not a track id"):
        paths.track_file(bad)


def test_track_file_takes_catalogue_ids(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert paths.track_file("marina_bay") == tmp_path / "tracks" / "marina_bay.track.json"

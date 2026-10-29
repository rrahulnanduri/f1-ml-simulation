import io
import re
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from f1sim import paths
from f1sim.data import tum_source as tum

FIXTURES = Path(__file__).parent / "fixtures" / "tum"
TRACK_FIXTURE = FIXTURES / "tracks" / "Silverstone.csv"
RACELINE_FIXTURE = FIXTURES / "racelines" / "Silverstone.csv"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    return tmp_path


class FakeUrlopen:
    """Serves fixture files for raw.githubusercontent.com URLs and counts calls."""

    def __init__(self, payload: bytes | None = None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.urls: list[str] = []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        if self.payload is not None:
            return io.BytesIO(self.payload)
        kind, name = url.rsplit("/", 2)[-2:]
        return io.BytesIO((FIXTURES / kind / name).read_bytes())


# ---------------------------------------------------------------- constants


def test_pinned_commit_is_full_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", tum.TUM_COMMIT)
    assert tum.TUM_REPO == "TUMFTM/racetrack-database"


def test_track_lists():
    assert len(tum.TUM_TRACKS) == len(set(tum.TUM_TRACKS)) == 25
    assert list(tum.TUM_TRACKS) == sorted(tum.TUM_TRACKS)
    assert {"Silverstone", "YasMarina", "Spa", "Zandvoort"} <= set(tum.TUM_TRACKS)
    assert set(tum.TUM_RACELINES) <= set(tum.TUM_TRACKS)


def test_url():
    assert tum.tum_url("tracks", "Spa") == (
        f"https://raw.githubusercontent.com/TUMFTM/racetrack-database/{tum.TUM_COMMIT}"
        "/tracks/Spa.csv"
    )


# ---------------------------------------------------------------- parsing


def test_parse_track_fixture():
    t = tum.parse_tum_csv(TRACK_FIXTURE)
    assert isinstance(t, tum.TumTrack)
    for arr in t:
        assert arr.dtype == np.float64
        assert arr.shape == (20,)
        assert arr.flags.c_contiguous
    assert t.x[0] == pytest.approx(3.439354)
    assert t.y[0] == pytest.approx(-0.495322)
    assert t.w_right[0] == pytest.approx(6.556)
    assert t.w_left[0] == pytest.approx(6.536)
    assert np.all((t.w_right > 2) & (t.w_right < 20))
    # Points are about 5 m apart
    step = np.hypot(np.diff(t.x), np.diff(t.y))
    assert np.all((step > 3) & (step < 7))


def test_parse_raceline_fixture():
    x, y = tum.parse_tum_raceline(RACELINE_FIXTURE)
    assert x.dtype == y.dtype == np.float64
    assert x.shape == y.shape == (20,)
    assert (x[0], y[0]) == pytest.approx((-1.227574, 2.890894))


def test_parse_without_header(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("0,0,5,5\n5,0,5,5\n5,5,5,5\n")
    t = tum.parse_tum_csv(p)
    assert t.x.tolist() == [0, 5, 5]


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("# x_m,y_m\n0,0\n1,1\n2,2\n", "header"),
        ("# x_m,y_m,w_tr_right_m,w_tr_left_m\n0,0,1\n1,1,1\n2,2,1\n", "columns"),
        ("# x_m,y_m,w_tr_right_m,w_tr_left_m\n0,0,1,1\n1,1,1,1\n", "at least 3"),
        ("# x_m,y_m,w_tr_right_m,w_tr_left_m\n0,0,1,1\n1,nan,1,1\n2,2,1,1\n", "non-finite"),
        ("# x_m,y_m,w_tr_right_m,w_tr_left_m\n0,0,1,1\n1,1,-1,1\n2,2,1,1\n", "negative"),
    ],
)
def test_parse_track_rejects_bad_files(tmp_path, text, match):
    p = tmp_path / "bad.csv"
    p.write_text(text)
    with pytest.raises(ValueError, match=match):
        tum.parse_tum_csv(p)


def test_raceline_parser_rejects_track_file():
    with pytest.raises(ValueError, match="header"):
        tum.parse_tum_raceline(TRACK_FIXTURE)


# ---------------------------------------------------------------- download + cache (offline)


def test_fetch_downloads_once_into_commit_folder(cache, monkeypatch):
    fake = FakeUrlopen()
    monkeypatch.setattr(tum.urllib.request, "urlopen", fake)
    p = tum.fetch_tum_csv("Silverstone")
    assert p == cache / "tum" / tum.TUM_COMMIT / "tracks" / "Silverstone.csv"
    assert p.read_bytes() == TRACK_FIXTURE.read_bytes()
    assert fake.urls == [tum.tum_url("tracks", "Silverstone")]
    assert tum.fetch_tum_csv("Silverstone") == p
    assert len(fake.urls) == 1  # served from the cache
    assert [f.name for f in p.parent.iterdir()] == ["Silverstone.csv"]  # no temp files left

    r = tum.fetch_tum_raceline("Silverstone")
    assert r == cache / "tum" / tum.TUM_COMMIT / "racelines" / "Silverstone.csv"
    assert tum.parse_tum_raceline(r)[0].shape == (20,)


@pytest.mark.parametrize("name", ["silverstone", "Monaco", "../Silverstone", ""])
def test_fetch_rejects_unknown_names(cache, monkeypatch, name):
    fake = FakeUrlopen()
    monkeypatch.setattr(tum.urllib.request, "urlopen", fake)
    with pytest.raises(ValueError, match="Unknown TUM"):
        tum.fetch_tum_csv(name)
    with pytest.raises(ValueError, match="Unknown TUM"):
        tum.fetch_tum_raceline(name)
    assert fake.urls == []


def test_unknown_name_suggests_close_match(cache):
    with pytest.raises(ValueError, match="Did you mean Silverstone"):
        tum.fetch_tum_csv("silverstone")


@pytest.mark.parametrize(
    "fake",
    [
        FakeUrlopen(error=urllib.error.URLError("offline")),
        FakeUrlopen(error=TimeoutError("timed out")),
        FakeUrlopen(payload=b"<html>404</html>"),
    ],
)
def test_failed_download_leaves_no_file(cache, monkeypatch, fake):
    monkeypatch.setattr(tum.urllib.request, "urlopen", fake)
    with pytest.raises(OSError, match="Spa"):
        tum.fetch_tum_csv("Spa")
    folder = cache / "tum" / tum.TUM_COMMIT / "tracks"
    assert not folder.exists() or list(folder.iterdir()) == []


# ---------------------------------------------------------------- network


@pytest.mark.network
def test_real_download_silverstone(cache):
    t = tum.parse_tum_csv(tum.fetch_tum_csv("Silverstone"))
    assert t.x.size > 1000
    closed_x = np.append(t.x, t.x[0])
    closed_y = np.append(t.y, t.y[0])
    length = np.hypot(np.diff(closed_x), np.diff(closed_y)).sum()
    assert 5800 < length < 6000  # official 5891 m
    width = t.w_left + t.w_right
    assert np.all((width > 8) & (width < 30))
    # The fixture is the head of the real file
    assert np.array_equal(t.x[:20], tum.parse_tum_csv(TRACK_FIXTURE).x)

    x, y = tum.parse_tum_raceline(tum.fetch_tum_raceline("Silverstone"))
    assert x.size > 1000 and x.size == y.size


def test_fetch_refuses_oversized_download(tmp_path, monkeypatch):
    monkeypatch.setenv("F1SIM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tum, "_MAX_BYTES", 16)
    monkeypatch.setattr(tum.urllib.request, "urlopen", FakeUrlopen(payload=b"# x_m\n" + b"1" * 64))
    with pytest.raises(OSError, match="over 16 bytes"):
        tum.fetch_tum_csv("Spa")

from importlib.metadata import version

import f1sim


def test_installed_version_matches_package():
    assert version("f1sim") == f1sim.__version__

import pytest

from f1sim import cli


@pytest.mark.network
def test_reference_lap_on_silverstone_is_within_ten_percent_of_pole(capsys):
    # Needs Silverstone built (`f1sim build-track silverstone`) and the TUM raceline download
    assert cli.main(["reference-lap", "silverstone"]) == 0
    out = capsys.readouterr().out
    assert "on track     yes" in out
    pct = float(out.split("->")[1].split("%")[0])
    assert abs(pct) <= 10.0

"""The audit's annual-energy check fails a series that measures part of the fleet.

Belgium's ENTSO-E files held onshore generation only in 2015-19 and 2023, with
onshore capacity as the denominator, so every physical gate passed. Only the
energy against a national statistic shows the missing offshore half.
"""

from __future__ import annotations

import pandas as pd
from support import ROOT, load_script


def _module():
    return load_script(ROOT / "scripts/analysis/audit_country_observations.py", "audit_obs")


def hourly(year: int, mw: float, hours: int | None = None) -> pd.DataFrame:
    index = pd.date_range(f"{year}-01-01", periods=hours or 8760, freq="1h", tz="UTC")
    return pd.DataFrame({"generation_mw": mw, "capacity_factor": 0.3}, index=index)


REFERENCE = pd.DataFrame({"country": ["BE"], "year": [2023], "gwh": [8760.0]})


def test_a_complete_series_passes():
    module = _module()
    ratios = module.annual_energy_ratios(hourly(2023, 900.0), REFERENCE, "BE")
    assert round(ratios.loc[2023], 6) == 0.9
    assert module.energy_findings(ratios) == ([], [])


def test_a_series_missing_half_the_fleet_fails():
    module = _module()
    ratios = module.annual_energy_ratios(hourly(2023, 450.0), REFERENCE, "BE")
    failures, notes = module.energy_findings(ratios)
    assert failures == ["2023 energy is 0.45 of the national statistic"]
    assert notes == []


def test_a_modest_shortfall_is_noted_not_failed():
    module = _module()
    ratios = module.annual_energy_ratios(hourly(2023, 800.0), REFERENCE, "BE")
    assert module.energy_findings(ratios) == ([], ["2023 energy is 0.80 of the national statistic"])


def test_a_partial_year_is_not_compared():
    module = _module()
    ratios = module.annual_energy_ratios(hourly(2023, 450.0, hours=24 * 30), REFERENCE, "BE")
    assert ratios.empty


def test_half_hourly_steps_count_energy_once():
    module = _module()
    index = pd.date_range("2023-01-01", periods=2 * 8760, freq="30min", tz="UTC")
    obs = pd.DataFrame({"generation_mw": 1000.0, "capacity_factor": 0.3}, index=index)
    ratios = module.annual_energy_ratios(obs, REFERENCE, "BE")
    assert round(ratios.loc[2023], 6) == 1.0

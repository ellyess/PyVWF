"""Pin the monthly capacity-factor conversions that no other pin covers.

Nine places turn monthly energy into a capacity factor by the days or hours in
the month. Seven are covered by real-data pins: the bootstrap pins load every
scorecard row through the european, EIA, ONS, AEMO, EMI, CEN and CAMMESA
adapters, and the NZ, Argentina and Chile processing pins run those
transforms. The WindStats transform and the client-CSV adapter have no such
pin, so these cases pin their output on synthetic data before the day and
hour counts move into one helper.

The data span 2019 and 2020, so February appears with 28 and 29 days, and use
awkward values so that a change in the order of operations shows in the last
digit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pyvwf.datasets.windstats import windstats_monthly_cf
from pyvwf.sources.client_csv import ClientCsvTurbineSource

PINS = Path(__file__).resolve().parent / "data" / "pins" / "month_hours"
MONTHS = [(y, m) for y in (2019, 2020) for m in range(1, 13)]


def windstats_output() -> pd.DataFrame:
    rows = []
    for k, unit in enumerate(["W1", "W2 ", "W3"]):
        for i, (y, m) in enumerate(MONTHS):
            rows.append(
                {"ID": unit, "Year": y, "Month": m, "Output": 1.0e5 / 3.0 * (k + 1) + 7.3 * i}
            )
    capacity = pd.DataFrame({"ID": ["W1", "W2", "W3"], "capacity": [600.0, 1650.0 / 7.0, 0.0]})
    return windstats_monthly_cf(pd.DataFrame(rows), capacity, 2019, 2020)


def client_csv_output(tmp_path: Path, unit: str) -> pd.DataFrame:
    meta = pd.DataFrame(
        {
            "ID": ["A", "B"],
            "lon": [9.0, 9.5],
            "lat": [56.0, 56.2],
            "capacity": [3.0, 2.0 / 3.0] if unit == "mw" else [3000.0, 2000.0 / 3.0],
            "height": [100.0, 90.0],
            "diameter": [112.0, 90.0],
        }
    )
    gen = pd.DataFrame(
        [
            {"ID": uid, "year": y, "month": m, "energy": (1.0e3 / 7.0) * (j + 1) + 0.1 * i}
            for j, uid in enumerate(["A", "B"])
            for i, (y, m) in enumerate(MONTHS)
        ]
    )
    mp, gp = tmp_path / "meta.csv", tmp_path / "gen.csv"
    meta.to_csv(mp, index=False)
    gen.to_csv(gp, index=False)
    source = ClientCsvTurbineSource(
        mp, gp, country="DK", capacity_unit=unit, generation_column="energy"
    )
    return source.load_observations()


def _compare(got: pd.DataFrame, name: str) -> None:
    want = pd.read_csv(PINS / name, dtype={"ID": str}, float_precision="round_trip")
    got = got.astype({"ID": str}).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False, rtol=0, atol=0)


def test_windstats_monthly_cf():
    _compare(windstats_output(), "windstats_monthly_cf.csv")


def test_client_csv_kw(tmp_path):
    _compare(client_csv_output(tmp_path, "kw"), "client_csv_kw.csv")


def test_client_csv_mw(tmp_path):
    _compare(client_csv_output(tmp_path, "mw"), "client_csv_mw.csv")


def test_leap_february_is_29_days():
    out = windstats_output().set_index(["ID", "year"])
    ratio = out.loc[("W1", 2019), "obs_2"] / out.loc[("W1", 2020), "obs_2"]
    # Same capacity, outputs differ by 7.3 * 12; the day count is the rest.
    feb19 = (1.0e5 / 3.0 + 7.3 * 1) / (600.0 * 28 * 24.0)
    feb20 = (1.0e5 / 3.0 + 7.3 * 13) / (600.0 * 29 * 24.0)
    assert np.isclose(ratio, feb19 / feb20, rtol=1e-12)


if __name__ == "__main__":  # records the fixtures; run once, by hand
    import tempfile

    PINS.mkdir(parents=True, exist_ok=True)
    windstats_output().to_csv(PINS / "windstats_monthly_cf.csv", index=False)
    for unit in ("kw", "mw"):
        with tempfile.TemporaryDirectory() as d:
            client_csv_output(Path(d), unit).to_csv(PINS / f"client_csv_{unit}.csv", index=False)

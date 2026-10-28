"""Helpers the test files share: the synthetic Denmark region, fleets, scripts.

Until 2026-10-16 these lived in the test files that first needed them, and
eleven files imported from ``test_harness_driver`` or ``test_pipeline`` to
reach them, so a test module doubled as a library (#90). They live here now;
``conftest.py`` builds the ``synthetic_dk`` fixture from them.

- the synthetic Denmark region: ``write_era5`` and ``write_fleet`` write the
  on-disk layout the DK loaders read, with a planted bias (``TRUE_CF``), and
  ``make_spec`` is its region config;
- ``fleet_frame`` builds a fleet in the harness's schema, so a test names only
  the columns it cares about and a schema change is made once;
- ``load_script`` imports a script under ``scripts/`` by path.
"""

from __future__ import annotations

import importlib.util
from calendar import monthrange
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import xarray as xr

from pyvwf.harness.regions import RegionSpec

ROOT = Path(__file__).resolve().parents[1]

# The reanalysis spans these years; observations are generated for the same
# window so the training merge (on year+month) has something to join on.
YEARS = (2015, 2016)
YEAR_TEST = 2016

# The planted bias: turbines actually generate at this capacity factor, while
# the synthetic reanalysis winds are strong enough to simulate well above it.
# A correct bias correction must therefore scale the wind speeds *down*.
TRUE_CF = 0.25


def write_era5(era5_dir, seed=11):
    """Hourly u/v at 10 m and 100 m on a small grid inside the DK bounding box.

    A west-east wind gradient gives the two turbine clusters genuinely
    different biases, so a per-cluster correction has something to learn that a
    single global correction could not capture.
    """
    era5_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    times = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31 23:00", freq="h")
    lats = np.array([55.0, 55.5, 56.0])
    lons = np.array([8.0, 8.75, 9.5])
    shape = (len(times), len(lats), len(lons))

    # Windier in the west (low lon) than the east.
    gradient = np.linspace(1.5, -1.5, len(lons))[None, None, :]
    base = 10.0 + gradient + rng.normal(0.0, 1.5, size=shape)
    base = np.clip(base, 0.5, None)

    # Split into u/v components; 10 m winds follow a log profile with z0 ~ 0.05 m.
    u100 = base / np.sqrt(2.0)
    v100 = base / np.sqrt(2.0)
    shear = np.log(10 / 0.05) / np.log(100 / 0.05)
    u10, v10 = u100 * shear, v100 * shear

    xr.Dataset(
        {
            "u100": (("time", "lat", "lon"), u100),
            "v100": (("time", "lat", "lon"), v100),
            "u10": (("time", "lat", "lon"), u10),
            "v10": (("time", "lat", "lon"), v10),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    ).to_netcdf(era5_dir / "era5_synthetic.nc")


def write_fleet(dk_dir):
    """Six onshore turbines in two spatial groups, in the DK loader's schema.

    Manufacturer/capacity/diameter give a p_density of ~398, so `add_models`
    resolves them to the nearest bundled open-library curve
    (`NREL_Reference_5MW_126`, p_density ~401).
    """
    dk_dir.mkdir(parents=True, exist_ok=True)

    west = [(55.1, 8.1), (55.4, 8.2), (55.2, 8.3)]
    east = [(55.6, 9.3), (55.9, 9.4), (55.7, 9.2)]
    rows = []
    for i, (lat, lon) in enumerate(west + east):
        rows.append(
            {
                "ID": f"t{i}",
                "manufacturer": "Synthetic",
                "capacity": 2000.0,  # kW
                "diameter": 80.0,
                "height": 100.0,
                "lon": lon,
                "lat": lat,
                "location_type": "Land",
            }
        )
    fleet = pd.DataFrame(rows)
    fleet.to_csv(dk_dir / "dk_md.csv", index=False)

    # Monthly generation (kWh) consistent with TRUE_CF, which is how
    # prep_country inverts it: cf = generation / (days * 24 * capacity_kW).
    obs = []
    for turb in fleet.itertuples():
        for year in YEARS:
            for month in range(1, 13):
                hours = monthrange(year, month)[1] * 24
                obs.append(
                    {
                        "ID": turb.ID,
                        "year": year,
                        "month": month,
                        "generation_kwh": TRUE_CF * hours * turb.capacity,
                    }
                )
    pd.DataFrame(obs).to_csv(dk_dir / "dk_obs_2002_2020.csv", index=False)
    return fleet


NH = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


# The box the removed BoundingBoxes table gave Denmark, which the synthetic
# grid lies inside; tests pass it explicitly as a region config would.
DK_BBOX = (7.5, 13.5, 54.0, 58.2)


def make_spec(**overrides) -> RegionSpec:
    base = dict(
        code="DK",
        name="Denmark (synthetic)",
        source="european-turbine",
        obs_level="turbine",
        obs_unit="turbine",
        train_years=(2015, 2015),
        test_years=(2016,),
        era5_path="era5",
        bbox=DK_BBOX,
        file_tag="SYN",
        correction_model="affine-wind",
        cluster_list=(2,),
        time_slices=("fixed",),
        seasons=NH,
    )
    base.update(overrides)
    return RegionSpec(**base)


#: The harness's fleet schema, with values a test that does not care can take.
FLEET_DEFAULTS = {
    "lat": 55.5,
    "lon": 8.5,
    "height": 100.0,
    "capacity": 2000.0,
    "model": "GE.1.5sle",
    "type": "onshore",
}


def fleet_frame(n: int | None = None, **columns) -> pd.DataFrame:
    """A fleet in the harness's schema: ``ID`` plus ``FLEET_DEFAULTS``.

    Args:
        n: Number of units. Taken from the longest column given when omitted.
        **columns: Columns to set, each a scalar or a sequence of length ``n``.
            Anything not given takes its default; ``ID`` defaults to
            ``t0``, ``t1``, ...

    Returns:
        DataFrame with ``ID``, every default column and any extra column given,
        in that order.
    """
    lengths = [len(v) for v in columns.values() if not np.isscalar(v) and v is not None]
    n = n if n is not None else (max(lengths) if lengths else 1)
    frame = {"ID": columns.pop("ID", [f"t{i}" for i in range(n)])}
    for name, default in FLEET_DEFAULTS.items():
        frame[name] = columns.pop(name, default)
    frame.update(columns)
    return pd.DataFrame({k: (v if not np.isscalar(v) else [v] * n) for k, v in frame.items()})


def two_cluster_fleet(cluster=(0, 0, 1, 1)) -> pd.DataFrame:
    """Four units inside the conftest grid, two per cluster by default.

    Unequal capacities within the first cluster, so a capacity-weighted mean
    and an unweighted one differ there.
    """
    return fleet_frame(
        ID=["a", "b", "c", "d"],
        lat=[55.2, 55.4, 55.6, 55.8],
        lon=[8.2, 8.4, 8.6, 8.8],
        capacity=[1000.0, 3000.0, 2000.0, 2000.0],
        cluster=list(cluster),
    )


def load_script(relpath: str | Path, name: str | None = None) -> ModuleType:
    """Import a script by path, relative to the repository root or absolute.

    Scripts under ``scripts/`` are not a package, so a test that exercises one
    loads it by file. ``name`` is the module name; the file's stem by default.
    """
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name or path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

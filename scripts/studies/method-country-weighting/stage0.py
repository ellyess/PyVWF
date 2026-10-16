"""Stage 0 of the country weighting study (#67): one row, no fitting.

For each training year, the uncorrected national monthly capacity factor is
computed twice from the same simulation: weighted by condition A's grid (the
end-of-year grid of the last training year, which the scorecard run reads for
every training year) and by that year's mid-year grid. Nothing is fitted and
no observation is read beyond what ``train_set`` loads to align months.
Registered in ``docs/findings/method-country-weighting-prereg.md``, which fixes
the kill rule this output is read against.

The simulation is the harness's own: ``train_set`` with the row's scorecard
config, as ``run_train`` calls it, so each grid point's monthly capacity factor
is exactly the one a training run fits from. The two conditions differ only in
the capacity each point carries, and the mid-year grids are checked to hold
the same points, clusters, models, positions and heights as A's grid.

Usage, from the repository root, one row per process, through the run lock:

    PYVWF_INPUT=input/combined python scripts/dev/run_locked.py -- \\
        python scripts/studies/method-country-weighting/stage0.py \\
        configs/regions/scorecard/fr_country.toml output/country_weighting_2026-10-16/stage0
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pyvwf.cli.common import make_parser
from pyvwf.data import train_set
from pyvwf.config import PyVWFPaths
from pyvwf.harness.driver import era5_dir, resolve_source
from pyvwf.harness.regions import load_region
from pyvwf.metrics import weighted_mean
from pyvwf.provenance import build_manifest, write_manifest

#: Registered thresholds: a row clears Stage 0 only if both are met.
MIN_SCALAR_SHIFT, MIN_RMS = 0.01, 0.005
#: Columns that must not change between A's grid and a mid-year grid.
FIXED = ["ID", "cluster", "model", "lon", "lat", "height"]
#: The registered checksums of the end-of-year grids condition A reads.
A_CHECKSUMS = Path("output/offshore_regen_2026-10-16/after.sha256")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_dir(code: str) -> Path:
    return PyVWFPaths.COUNTRY_LEVEL_DATA / "grid_points" / code.lower()


def checked_grid_a(code: str, last_year: int) -> Path:
    """A's grid by its registered name, refused unless its checksum is listed.

    Built explicitly rather than taken from the source's resolver, which falls
    back to the static grid when a year's file is missing.
    """
    path = grid_dir(code) / f"{code.lower()}_grid_points_{last_year}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{code}: A's grid {path} does not exist")
    listed = dict(reversed(line.split(maxsplit=1)) for line in A_CHECKSUMS.read_text().splitlines())
    if listed.get(path.name) != _sha256(path):
        raise ValueError(
            f"{code}: {path.name} is not the grid registered in {A_CHECKSUMS}; "
            "condition A would not be the scorecard's run."
        )
    return path


def midyear_path(code: str, year: int) -> Path:
    """The mid-year grid of ``year`` beside the end-of-year ones."""
    return grid_dir(code) / f"{code.lower()}_grid_points_{year}_midyear.csv"


def left_out(code: str) -> dict[int, float]:
    """MW the builder could not place on the fixed point set, per year."""
    path = grid_dir(code) / f"{code.lower()}_midyear_left_out.json"
    return {int(k): float(v) for k, v in json.loads(path.read_text()).items()}


def check_same_points(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    """Refuse unless only capacity differs between two grids."""
    left = a[FIXED].assign(ID=a["ID"].astype(str)).sort_values("ID").reset_index(drop=True)
    right = b[FIXED].assign(ID=b["ID"].astype(str)).sort_values("ID").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(
            f"{label}: the mid-year grid differs from A's grid in a column other than "
            "capacity, so the two conditions would not simulate the same sites."
        )


def national(sim: pd.DataFrame, capacity: pd.Series) -> pd.Series:
    """Capacity-weighted national CF per (year, month), as scoring computes it."""
    caps = capacity.reindex(sim.columns).to_numpy(dtype=float)
    return pd.Series(weighted_mean(sim.to_numpy(), caps[None, :], axis=1), index=sim.index)


def summarise(monthly: pd.DataFrame) -> dict:
    both = monthly.dropna(subset=["cf_a", "cf_b"])
    rms = float(np.sqrt(np.mean((both["cf_b"] - both["cf_a"]) ** 2)))
    shift = float(abs(both["cf_b"].mean() / both["cf_a"].mean() - 1.0))
    return {
        "months_compared": int(len(both)),
        "months_dropped": int(len(monthly) - len(both)),
        "rms_difference": rms,
        "scalar_shift": shift,
        "ratio_midyear_to_a": float(both["cf_b"].mean() / both["cf_a"].mean()),
        "clears": bool(shift >= MIN_SCALAR_SHIFT and rms >= MIN_RMS),
    }


def extremes(grid_a: pd.DataFrame, mids: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """The smallest and largest points in A's grid, followed through each year."""
    cap_a = grid_a.set_index("ID")["capacity"]
    positive = cap_a[cap_a > 0]
    picks = {"smallest": positive.idxmin(), "largest": cap_a.idxmax()}
    rows = []
    for label, pid in picks.items():
        for year, mid in mids.items():
            cap_mid = mid.set_index("ID")["capacity"]
            rows.append(
                {
                    "point": label,
                    "ID": pid,
                    "cluster": int(grid_a.set_index("ID").loc[pid, "cluster"]),
                    "year": year,
                    "capacity_a_mw": float(cap_a[pid]),
                    "capacity_midyear_mw": float(cap_mid[pid]),
                    "share_a": float(cap_a[pid] / cap_a.sum()),
                    "share_midyear": float(cap_mid[pid] / cap_mid.sum()),
                }
            )
    return pd.DataFrame(rows)


def zero_capacity(grid_a: pd.DataFrame, mids: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Every point, and every cluster, with no capacity in A's grid or a year's."""
    rows = []
    grids = {"A": grid_a, **{str(year): mid for year, mid in mids.items()}}
    for label, grid in grids.items():
        cap = pd.to_numeric(grid["capacity"], errors="coerce")
        for row in grid.loc[~(cap > 0), ["ID", "cluster"]].itertuples():
            rows.append({"grid": label, "level": "point", "ID": row.ID, "cluster": row.cluster})
        by_cluster = cap.groupby(grid["cluster"]).sum()
        for cluster in by_cluster.index[~(by_cluster > 0)]:
            rows.append({"grid": label, "level": "cluster", "ID": "", "cluster": cluster})
    return pd.DataFrame(rows, columns=["grid", "level", "ID", "cluster"])


def main(config: str, out_dir: str) -> None:
    spec = load_region(config)
    if spec.obs_level != "country":
        raise ValueError(f"{spec.code}: Stage 0 is defined for country-level rows only")
    if Path(PyVWFPaths.INPUT_ROOT).resolve() != Path("input/combined").resolve():
        raise ValueError(
            f"input root is {PyVWFPaths.INPUT_ROOT}; the registration names input/combined"
        )
    out = Path(out_dir) / spec.code
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds a Stage 0 result")
    first, last = spec.train_years
    years = list(range(first, last + 1))
    grid_a_path = checked_grid_a(spec.code, last)
    source = resolve_source(spec)

    gen_cf, turb_info, _reanalysis, curves = train_set(
        spec.code,
        True,
        "all",
        obs_level=spec.obs_level,
        source=source,
        era5_dir=era5_dir(spec),
        bbox=spec.bbox,
        allow_extrapolation=spec.allow_extrapolation,
        roughness=spec.roughness,
    )

    grid_a = source.load_metadata()
    if not grid_a[FIXED].equals(
        pd.read_csv(grid_a_path).assign(ID=lambda f: f["ID"].astype(str))[FIXED]
    ):
        raise ValueError(f"{spec.code}: the source did not load A's registered grid")
    missing_curves = sorted(set(turb_info["model"].astype(str)) - set(curves.columns))
    if missing_curves:
        raise ValueError(f"{spec.code}: no curve in the library for {missing_curves}")
    mids: dict[int, pd.DataFrame] = {}
    grid_files = {grid_a_path.name: _sha256(grid_a_path)}
    for year in years:
        path = midyear_path(spec.code, year)
        if not path.is_file():
            raise FileNotFoundError(f"{spec.code}: no mid-year grid for {year} at {path}")
        mid = pd.read_csv(path).assign(ID=lambda f: f["ID"].astype(str))
        check_same_points(grid_a, mid, f"{spec.code} {year}")
        mids[year] = mid
        grid_files[path.name] = _sha256(path)

    sim = gen_cf.pivot_table(index=["year", "month"], columns="ID", values="sim", aggfunc="first")
    sim = sim[sim.index.get_level_values("year").isin(years)]
    if set(sim.columns) != set(grid_a["ID"]):
        raise ValueError(f"{spec.code}: the simulated points are not A's grid points")

    cf_a = national(sim, grid_a.set_index("ID")["capacity"])
    cf_b = pd.concat(
        [
            national(sim.loc[[year]], mids[year].set_index("ID")["capacity"])
            for year in years
            if year in sim.index.get_level_values("year")
        ]
    )
    monthly = pd.DataFrame({"cf_a": cf_a, "cf_b": cf_b}).reset_index()

    unplaced = left_out(spec.code)
    totals = pd.DataFrame(
        {
            "year": years,
            "grid_total_a_mw": float(grid_a["capacity"].sum()),
            "grid_total_midyear_mw": [float(mids[y]["capacity"].sum()) for y in years],
            "midyear_left_out_mw": [unplaced.get(y, 0.0) for y in years],
        }
    )
    summary = {
        "region": spec.code,
        "train_years": [first, last],
        "grid_a": grid_a_path.name,
        "cluster_list": list(spec.cluster_list),
        "clusters_in_grid": int(grid_a["cluster"].nunique()),
        "thresholds": {"scalar_shift": MIN_SCALAR_SHIFT, "rms_difference": MIN_RMS},
        **summarise(monthly),
        "per_year": {str(y): summarise(monthly[monthly["year"] == y]) for y in years},
    }

    out.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out / "monthly.csv", index=False)
    totals.to_csv(out / "grid_totals.csv", index=False)
    extremes(grid_a, mids).to_csv(out / "extremes.csv", index=False)
    zero_capacity(grid_a, mids).to_csv(out / "zero_capacity.csv", index=False)
    (out / "stage0.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(
        out,
        build_manifest(
            spec,
            extra={
                "run_mode": "stage0",
                "study": "method-country-weighting",
                "grid_files_sha256": grid_files,
            },
        ),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_year"}, indent=2))


def cli(argv: list[str] | None = None) -> None:
    parser = make_parser("Stage 0 of the country weighting study, one row.")
    parser.add_argument("config", help="the row's scorecard config")
    parser.add_argument("out_dir", help="the study's stage0 directory")
    args = parser.parse_args(argv)
    main(args.config, args.out_dir)


if __name__ == "__main__":
    cli()

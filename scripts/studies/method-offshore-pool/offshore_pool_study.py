"""The offshore pool study: declared modes against shape classification.

Registered in ``docs/findings/method-offshore-pool-prereg.md``, committed before
it runs. Thesis chapter 4 reports that gridded kriging makes Denmark offshore
worse than no correction at all and gives a cause, "only 2 offshore control
points". That count comes from the declared ``cluster_mode``; classifying the
same points against the region shapes gives Denmark 13. This measures whether
the failure survives the larger pool.

Everything the registration fixes is fixed here rather than on the command line:

- **P0** interpolates the offshore surface from the 12 declared-offshore
  control points and the onshore surface from the other 1,717, which is what
  the chapter's tables did. **P1** uses the shape classification instead.
- A control point the shapes place in neither domain, ten of the 1,729, joins
  the onshore pool under P1. That mirrors the declared rule, which sends
  country-level points there, and it is stated because it is a choice.
- **The study-scoped bounding box widens Denmark to 15.4 east**, so the 47
  chapter-era units beyond `dk.toml`'s 13.5 are simulated from real winds
  rather than extrapolated ones. `dk.toml` is untouched.
- **The neutral-fill share is reported before any error metric**, for every row
  and both conditions. A row whose units mostly fall outside the surface has
  not been corrected, and its error would otherwise read as an ordinary result.
- **Today's uncorrected MAE is reported beside the chapter's published one**,
  per the cross-pipeline rule in the registration: gate O1 is stated against a
  figure from another pipeline and is verified against today's before it is
  read.

Read-only with respect to the tree. Writes its results under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/studies/method-offshore-pool/offshore_pool_study.py <out_dir>
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vwf.cli.common import make_parser
from vwf.datasets.era5 import prep_era5
from vwf.curves import load_power_curves
from vwf.extensions.grid import evaluate, surface
from vwf.geospatial import categorize_points_spatial_join
from vwf.wind import interpolate_wind

POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")
RUNS = Path("output/runs/turbine_grid")
SHAPES = Path("input/reference/shapes")

#: The chapter's target grid.
GRID_LON = np.arange(-10.0, 30.01, 0.25)
GRID_LAT = np.arange(35.0, 72.01, 0.25)

#: Study-scoped boxes. Denmark's east edge is 15.4 rather than dk.toml's 13.5,
#: to cover the chapter-era fleet without extrapolating; see the registration.
BBOX = {"DK": (7.5, 15.4, 54.0, 58.2), "UK": (-11.0, 3.0, 49.0, 61.0)}

#: The rows whose control-point membership differs between the two conditions,
#: with their test year and the cluster count the chapter reports for them.
ROWS = (
    ("DK", "offshore", 2020, 2),
    ("DK", "onshore", 2020, 700),
    ("UK", "offshore", 2019, 10),
    ("UK", "onshore", 2019, 300),
)

#: What the chapter published for these rows, for the cross-pipeline check.
PUBLISHED_UNCORRECTED = {
    "DK offshore": 0.0822,
    "DK onshore": 0.129,
    "UK offshore": 0.160,
    "UK onshore": 0.081,
}


def run_dir(code: str, mode: str, runs: Path = RUNS) -> Path:
    return Path(runs) / f"{code}-{mode}-obs_turbine-corrected-calc_z0"


def load_row(code: str, mode: str, year: int, clusters: int, runs: Path = RUNS) -> dict:
    """The chapter-era fleet, observations and cluster-based comparison."""
    base = run_dir(code, mode, runs)
    fleet = pd.read_csv(base / "training" / "simulated-turbines" / f"{code}_{year}_turb_info.csv")
    results = base / "results" / "capacity-factor"
    return {
        "fleet": fleet,
        "observed": pd.read_csv(results / f"{code}_{year}_obs_cf.csv"),
        "cluster_cf": pd.read_csv(results / f"{code}_{year}_fixed_{clusters}_cor_cf.csv"),
        "uncorrected_cf": pd.read_csv(results / f"{code}_{year}_unc_cf.csv"),
    }


def pools(pool: pd.DataFrame, shapes: Path = SHAPES) -> dict[str, pd.DataFrame]:
    """The two conditions, as two frames differing only in their domain column.

    P1's ``unknown`` points, which the shapes place in neither domain, join
    onshore. The declared rule does the same with country-level points.
    """
    p0 = pool.assign(domain=surface.declared_domains(pool, domain_col="cluster_mode"))
    by_shape = categorize_points_spatial_join(
        pool,
        onshore_geojson=Path(shapes) / "country_shapes.geojson",
        offshore_geojson=Path(shapes) / "offshore_shapes.geojson",
    )
    p1 = pool.assign(domain=np.where(by_shape.to_numpy() == "offshore", "offshore", "onshore"))
    return {"P0": p0, "P1": p1}


def main(out_dir: str, pool_path: Path = POOL, runs: Path = RUNS, shapes: Path = SHAPES) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onshore = Path(shapes) / "country_shapes.geojson"
    offshore = Path(shapes) / "offshore_shapes.geojson"
    pool = pd.read_csv(pool_path)
    conditions = pools(pool, shapes)
    for name, frame in conditions.items():
        counts = frame["domain"].value_counts().to_dict()
        print(f"{name}: onshore {counts.get('onshore', 0)}, offshore {counts.get('offshore', 0)}")

    surfaces = {}
    for name, frame in conditions.items():
        print(f"\nbuilding the {name} surface ...", flush=True)
        surfaces[name] = surface.correction_surface(
            frame,
            GRID_LON,
            GRID_LAT,
            onshore_geojson=onshore,
            offshore_geojson=offshore,
            domain_col="domain",
            method="kriging",
        )
        print(
            f"  {name}: {surfaces[name].attrs['n_control_points_onshore']} onshore, "
            f"{surfaces[name].attrs['n_control_points_offshore']} offshore",
            flush=True,
        )

    curves = load_power_curves()
    rows = []
    for code, mode, year, clusters in ROWS:
        label = f"{code} {mode}"
        print(f"\n=== {label}", flush=True)
        data = load_row(code, mode, year, clusters, runs)
        fleet = data["fleet"]
        reanalysis = prep_era5(code, False, True, bbox=BBOX[code], era5_dir=None)
        reanalysis = reanalysis.sel(time=str(year))
        speed = interpolate_wind(reanalysis, fleet)

        base = {"row": label, "year": year, "units": len(fleet)}
        # Today's uncorrected, the cross-pipeline check, before anything else.
        neutral = pd.DataFrame(
            {"ID": fleet["ID"].astype(str), "scalar": 1.0, "offset": 0.0, "neutral": True}
        )
        unc_cf, unc_summary = evaluate.corrected_capacity_factors(speed, neutral, curves)
        unc = evaluate.turbine_skill(unc_cf, data["observed"], fleet)
        rows.append(
            {
                **base,
                "condition": "uncorrected today",
                **unc,
                "off_curve_share": unc_summary["off_curve_share"],
                "published_uncorrected": PUBLISHED_UNCORRECTED.get(label),
            }
        )

        cluster = evaluate.turbine_skill(data["cluster_cf"], data["observed"], fleet)
        rows.append({**base, "condition": f"cluster fixed_{clusters}", **cluster})

        for name, field in surfaces.items():
            corrections, summary = evaluate.corrections_at(field, fleet)
            cf, cf_summary = evaluate.corrected_capacity_factors(speed, corrections, curves)
            skill = evaluate.turbine_skill(cf, data["observed"], fleet)
            rows.append(
                {
                    **base,
                    "condition": f"grid kriging {name}",
                    **skill,
                    **summary,
                    "off_curve_share": cf_summary["off_curve_share"],
                }
            )
            print(
                f"  {name}: neutral {summary['neutral_share']:.1%} "
                f"({summary['n_neutral']} of {summary['n_units']}), "
                f"MAE {skill['mae']:.4f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "offshore_pool_results.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n=== neutral fill, before any metric")
        print(
            frame[frame["condition"].str.startswith("grid")][
                ["row", "condition", "n_units", "n_neutral", "neutral_share", "n_off_grid"]
            ]
            .round(4)
            .to_string(index=False)
        )
        print("\n=== today's uncorrected against the chapter's published")
        print(
            frame[frame["condition"] == "uncorrected today"][
                ["row", "mae", "published_uncorrected", "off_curve_share"]
            ]
            .round(4)
            .to_string(index=False)
        )
        print("\n=== all conditions")
        print(frame[["row", "condition", "mae", "rmse", "bias"]].round(4).to_string(index=False))
    print(f"\nwritten: {out / 'offshore_pool_results.csv'}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument(
        "--pool", type=Path, default=POOL, help=f"The control-point pool (default: {POOL})"
    )
    parser.add_argument(
        "--runs", type=Path, default=RUNS, help=f"The chapter's thesis-era runs (default: {RUNS})"
    )
    parser.add_argument(
        "--shapes",
        type=Path,
        default=SHAPES,
        help=f"The onshore and offshore GeoJSON directory (default: {SHAPES})",
    )
    args = parser.parse_args(argv)
    main(args.out_dir, pool_path=args.pool, runs=args.runs, shapes=args.shapes)


if __name__ == "__main__":
    cli()

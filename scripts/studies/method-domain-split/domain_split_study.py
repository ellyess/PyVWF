"""Does correcting a unit from control points of a different kind cause the failure?

Registered in ``docs/findings/method-domain-split-prereg.md``, committed before
it runs. Thesis chapter 4 interpolates one surface from all 1,729 control
points, so an offshore unit takes a correction dominated by whatever is nearest,
which for Denmark's offshore sites is 884 Danish onshore clusters. Denmark
offshore's self-weight is 0.045, it is the only configuration of the fourteen
that is not majority its own answer, and it is the only one the chapter reports
as failing.

Three conditions, because the split as first tested changed two things at once:

- **S0** undivided pool, distance mask. The chapter's own arrangement.
- **S1** split by declared ``cluster_mode``, distance mask. **Isolates the
  pool**, which is the question.
- **S2** split, area-of-interest mask. What the port does; reported so the
  port's behaviour is on the record, and it decides nothing.

All fourteen configurations are scored, not the four whose membership changes.
A configuration whose pool does not change should not move, and checking that is
how the comparison is verified rather than assumed.

**Every row runs on real winds.** The 2026-09-15 amendment to the registration
moved the study from ``era5/EU``, which stops at 42 north, to
``era5/EU_2026-09``, which reaches 36 north and covers every unit of every
configuration. Five rows could not otherwise be simulated without extrapolating
past the data, and their published figures were produced that way. The two
archives carry bit-identical winds where they overlap and differ in one other
respect: the older files carry a stored annual-mean roughness and the newer ones
carry none, so the newer route derives roughness per timestep, which is the
method this project adopted on 2026-09-12. **The nine rows that need neither
change therefore also run a fourth condition on the chapter's own archive**, so
the roughness treatment is measured rather than assumed when S-G1 is read.

Read-only with respect to the tree. Writes its results under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/studies/method-domain-split/domain_split_study.py <out_dir>
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pyvwf.cli.common import make_parser
from pyvwf.config import BoundingBoxes
from pyvwf.curves import load_power_curves
from pyvwf.era5 import prep_era5
from pyvwf.extensions.grid import evaluate, interpolation as interp, surface
from pyvwf.wind import interpolate_wind

POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")
RUNS = Path("output/runs/turbine_grid")
SHAPES = Path("input/reference/shapes")

#: The archive every row is simulated from: 36 to 72 north, 12 west to 31.5
#: east, hourly components and no stored roughness.
ERA5 = Path("input/era5/EU_2026-09")
#: The chapter's archive, 42 to 72 north, carrying an annual-mean roughness.
#: Used only for the reference condition on rows it covers.
ERA5_CHAPTER = Path("input/era5/EU")

GRID_LON = np.arange(-10.0, 30.01, 0.25)
GRID_LAT = np.arange(35.0, 72.01, 0.25)

#: Study-scoped box for Denmark, 15.4 east rather than dk.toml's 13.5, so the
#: chapter-era fleet is simulated from real winds. Every other configuration
#: uses its shipped box, which already covers its own units; what clipped the
#: five southern and eastern rows was the archive, not the box.
BBOX = {"DK": (7.5, 15.4, 54.0, 58.2)}

#: code, mode, obs_level, test year, the cluster count the chapter reports, and
#: the chapter's published grid kriging MAE, for gate S-G1.
CONFIGURATIONS = (
    ("BE", "all", "country", 2023, 3, 0.0192),
    ("DE", "onshore", "turbine", 2019, 500, 0.0411),
    ("DK", "offshore", "turbine", 2020, 2, 0.1113),
    ("DK", "onshore", "turbine", 2020, 700, 0.0678),
    ("ES", "all", "country", 2023, 4, 0.0255),
    ("FR", "all", "country", 2023, 10, 0.0301),
    ("IE", "all", "country", 2023, 3, 0.1563),
    ("IT", "all", "country", 2023, 3, 0.0308),
    ("NL", "all", "country", 2023, 5, 0.0563),
    ("NO", "all", "country", 2023, 5, 0.0381),
    ("PT", "all", "country", 2023, 3, 0.0392),
    ("SE", "all", "country", 2023, 4, 0.0357),
    ("UK", "offshore", "turbine", 2019, 10, 0.1338),
    ("UK", "onshore", "turbine", 2019, 300, 0.0616),
)

#: The share of each row's capacity that the chapter's archive does not reach,
#: measured on the chapter-era fleets. These five rows' published figures were
#: produced from winds extrapolated past the data, so S-G1 is reported for them
#: as a difference and not as a pass or fail. The other nine are zero and are
#: what the gate is read on.
CHAPTER_EXTRAPOLATED = {
    "ES all": 0.4280,
    "IT all": 0.8935,
    "NO all": 0.3568,
    "PT all": 0.8492,
    "SE all": 0.0546,
}

#: The declared pool's own name for a country-level control point.
POOL_CODE = {
    ("DK", "offshore"): "DK-offshore",
    ("DK", "onshore"): "DK-onshore",
    ("UK", "offshore"): "UK-offshore",
    ("UK", "onshore"): "UK-onshore",
    ("DE", "onshore"): "DE-onshore",
}


def pool_code(code: str, mode: str) -> str:
    return POOL_CODE.get((code, mode), code)


def self_weight(pool: pd.DataFrame, code: str, mode: str) -> float:
    """Share of the undivided pool's weight at this configuration's own cells
    that comes from its own control points. The mechanism's own variable."""
    own = pool["country_code"] == pool_code(code, mode)
    mine = pool[own]
    cells = np.unique(
        np.column_stack(
            [
                GRID_LON[
                    np.abs(mine["lon"].to_numpy()[:, None] - GRID_LON[None, :]).argmin(axis=1)
                ],
                GRID_LAT[
                    np.abs(mine["lat"].to_numpy()[:, None] - GRID_LAT[None, :]).argmin(axis=1)
                ],
            ]
        ),
        axis=0,
    )
    d = interp.degree_distances(cells, pool[["lon", "lat"]].to_numpy(float))
    w = 1.0 / np.where(d < 1e-12, 1e-12, d) ** 2
    return float(np.median(w[:, own.to_numpy()].sum(axis=1) / w.sum(axis=1)))


def distance_masked(scalar, offset, points: pd.DataFrame) -> xr.Dataset:
    """The chapter's mask: neutral beyond 5 degrees from any control point."""
    lon_grid, lat_grid = np.meshgrid(GRID_LON, GRID_LAT)
    far = interp.distance_to_nearest(
        points, lon_grid.ravel(), lat_grid.ravel(), metric="degrees"
    ).reshape(lat_grid.shape)
    keep = far <= interp.MAX_DISTANCE_DEG
    return xr.Dataset(
        {
            "scalar": (("lat", "lon"), np.where(keep, scalar, 1.0)),
            "offset": (("lat", "lon"), np.where(keep, offset, 0.0)),
        },
        coords={"lat": GRID_LAT, "lon": GRID_LON},
    )


def kriged(points: pd.DataFrame):
    return interp.to_grid(
        interp.kriging_at,
        points,
        GRID_LON,
        GRID_LAT,
        variogram_model=interp.KRIGING_VARIOGRAM,
        coordinates_type=interp.KRIGING_COORDINATES,
    )


def build_surfaces(pool: pd.DataFrame, shapes: Path = SHAPES) -> dict[str, xr.Dataset]:
    """S0, S1 and S2. S1 and S2 share their interpolation and differ in mask."""
    domain = surface.declared_domains(pool, domain_col="cluster_mode")
    on, off = pool[domain == "onshore"], pool[domain == "offshore"]

    print(f"S0: one surface from {len(pool)} points ...", flush=True)
    s0_scalar, s0_offset = kriged(pool)

    print(f"S1 and S2: {len(on)} onshore, {len(off)} offshore ...", flush=True)
    on_scalar, on_offset = kriged(on)
    off_scalar, off_offset = kriged(off)

    # S1 combines the two domain surfaces by the same area-of-interest rule S2
    # uses, then masks by distance instead. Without a rule for which surface a
    # cell takes, a split pool has no combined field at all; the mask is what
    # differs between the two conditions.
    on_area = surface.area_mask(
        GRID_LON, GRID_LAT, Path(shapes) / "country_shapes.geojson", name="on"
    ).values
    off_area = surface.area_mask(
        GRID_LON, GRID_LAT, Path(shapes) / "offshore_shapes.geojson", name="off"
    ).values
    split_scalar = np.where(on_area, on_scalar, np.where(off_area, off_scalar, np.nan))
    split_offset = np.where(on_area, on_offset, np.where(off_area, off_offset, np.nan))

    s2 = xr.Dataset(
        {
            "scalar": (("lat", "lon"), np.where(np.isnan(split_scalar), 1.0, split_scalar)),
            "offset": (("lat", "lon"), np.where(np.isnan(split_offset), 0.0, split_offset)),
        },
        coords={"lat": GRID_LAT, "lon": GRID_LON},
    )
    s1 = distance_masked(
        np.where(np.isnan(split_scalar), 1.0, split_scalar),
        np.where(np.isnan(split_offset), 0.0, split_offset),
        pool,
    )
    return {"S0": distance_masked(s0_scalar, s0_offset, pool), "S1": s1, "S2": s2}


def winds(code: str, box, year: int, fleet: pd.DataFrame, era5_dir: Path):
    """Daily speeds at the fleet, and what the archive actually supplied.

    ``allow_extrapolation`` is left at its default, so a unit outside the
    loaded extent stops the study rather than being simulated from winds that
    do not exist. The audit before the run says there are none.
    """
    reanalysis = prep_era5(code, False, True, bbox=box, era5_dir=era5_dir)
    detail = {
        "roughness": reanalysis.attrs.get("pyvwf_roughness_treatment"),
        "extent": (
            f"{float(reanalysis.lon.min()):g} to {float(reanalysis.lon.max()):g}, "
            f"{float(reanalysis.lat.min()):g} to {float(reanalysis.lat.max()):g}"
        ),
    }
    return interpolate_wind(reanalysis.sel(time=str(year)), fleet), detail


def main(
    out_dir: str,
    pool_path: Path = POOL,
    runs: Path = RUNS,
    shapes: Path = SHAPES,
    era5: Path = ERA5,
    era5_chapter: Path = ERA5_CHAPTER,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    era5, era5_chapter = Path(era5), Path(era5_chapter)
    pool = pd.read_csv(pool_path)
    surfaces = build_surfaces(pool, shapes)
    curves = load_power_curves()
    rows = []

    for code, mode, level, year, clusters, published in CONFIGURATIONS:
        label = f"{code} {mode}"
        print(f"\n=== {label}", flush=True)
        base = Path(runs) / f"{code}-{mode}-obs_{level}-corrected-calc_z0"
        fleet = pd.read_csv(
            base / "training" / "simulated-turbines" / f"{code}_{year}_turb_info.csv"
        )
        results = base / "results" / "capacity-factor"
        observed = pd.read_csv(results / f"{code}_{year}_obs_cf.csv")
        cluster_cf = pd.read_csv(results / f"{code}_{year}_fixed_{clusters}_cor_cf.csv")
        box = BBOX.get(code, BoundingBoxes.get(code))
        weight = self_weight(pool, code, mode)

        speed, detail = winds(code, box, year, fleet, era5)
        common = {
            "row": label,
            "obs_level": level,
            "year": year,
            "units": len(fleet),
            "self_weight": weight,
            "published_grid_kriging": published,
            "chapter_extrapolated_share": CHAPTER_EXTRAPOLATED.get(label, 0.0),
            "bbox": str(box),
            "archive": era5.name,
            **detail,
        }

        nothing = pd.DataFrame(
            {"ID": fleet["ID"].astype(str), "scalar": 1.0, "offset": 0.0, "neutral": True}
        )
        unc_cf, unc_off = evaluate.corrected_capacity_factors(speed, nothing, curves)
        rows.append(
            {
                **common,
                "condition": "uncorrected",
                **evaluate.skill(unc_cf, observed, fleet, level),
                "off_curve_share": unc_off["off_curve_share"],
            }
        )
        rows.append(
            {
                **common,
                "condition": f"cluster fixed_{clusters}",
                **evaluate.skill(cluster_cf, observed, fleet, level),
            }
        )

        for name, field in surfaces.items():
            corrections, summary = evaluate.corrections_at(field, fleet)
            cf, off = evaluate.corrected_capacity_factors(speed, corrections, curves)
            got = evaluate.skill(cf, observed, fleet, level)
            rows.append(
                {
                    **common,
                    "condition": name,
                    **got,
                    **summary,
                    "off_curve_share": off["off_curve_share"],
                }
            )
            print(
                f"  {name}: neutral {summary['neutral_share']:.1%}, "
                f"off curve {off['off_curve_share']:.1%}, MAE {got['mae']:.4f}",
                flush=True,
            )

        # The reference condition. Only the nine rows the chapter's own archive
        # covers can have one, and for them it separates the roughness
        # treatment from the pipeline when S-G1 is read.
        if label not in CHAPTER_EXTRAPOLATED:
            del speed
            chapter_speed, chapter_detail = winds(code, box, year, fleet, era5_chapter)
            corrections, summary = evaluate.corrections_at(surfaces["S0"], fleet)
            cf, off = evaluate.corrected_capacity_factors(chapter_speed, corrections, curves)
            got = evaluate.skill(cf, observed, fleet, level)
            rows.append(
                {
                    **common,
                    "archive": era5_chapter.name,
                    **chapter_detail,
                    "condition": "S0 chapter archive",
                    **got,
                    **summary,
                    "off_curve_share": off["off_curve_share"],
                }
            )
            print(f"  S0 chapter archive: MAE {got['mae']:.4f}", flush=True)
            del chapter_speed

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "domain_split_results.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== what each row was simulated from")
        print(
            frame.drop_duplicates(["row", "archive"])[
                ["row", "archive", "roughness", "bbox", "extent", "chapter_extrapolated_share"]
            ].to_string(index=False)
        )
        grids = frame[frame["condition"].isin(["S0", "S1", "S2", "S0 chapter archive"])]
        print("\n=== neutral fill and off curve, before any metric")
        print(
            grids[["row", "condition", "units", "n_neutral", "neutral_share", "off_curve_share"]]
            .round(4)
            .to_string(index=False)
        )

        s0 = frame[frame["condition"] == "S0"].copy()
        s0["difference"] = s0["mae"] - s0["published_grid_kriging"]
        gated = s0[s0["chapter_extrapolated_share"] == 0.0]
        print("\n=== S-G1, read on the nine rows the archive change does not touch")
        print(
            gated[["row", "mae", "published_grid_kriging", "difference"]]
            .round(4)
            .to_string(index=False)
        )
        print(f"  worst absolute difference: {gated['difference'].abs().max():.5f}")

        print("\n=== the five rows whose published figure rests on extrapolated winds")
        print("    reported as a difference, not as a pass or fail")
        moved = s0[s0["chapter_extrapolated_share"] > 0.0]
        print(
            moved[
                ["row", "chapter_extrapolated_share", "mae", "published_grid_kriging", "difference"]
            ]
            .round(4)
            .to_string(index=False)
        )

        reference = frame[frame["condition"] == "S0 chapter archive"]
        if not reference.empty:
            pair = (
                s0.set_index("row")["mae"]
                .to_frame("S0 new archive")
                .join(reference.set_index("row")["mae"].rename("S0 chapter archive"))
                .dropna()
            )
            pair["roughness effect"] = pair["S0 new archive"] - pair["S0 chapter archive"]
            print("\n=== the roughness treatment, measured on the nine")
            print(pair.round(5).to_string())

        print("\n=== all conditions, with self-weight")
        print(
            frame[["row", "self_weight", "condition", "mae", "rmse", "bias"]]
            .round(4)
            .to_string(index=False)
        )
    print(f"\nwritten: {out / 'domain_split_results.csv'}")


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
    parser.add_argument(
        "--era5",
        type=Path,
        default=ERA5,
        help=f"The ERA5 the surfaces are applied to (default: {ERA5})",
    )
    parser.add_argument(
        "--era5-chapter",
        type=Path,
        default=ERA5_CHAPTER,
        help=f"The chapter's own ERA5, for the archive comparison (default: {ERA5_CHAPTER})",
    )
    args = parser.parse_args(argv)
    main(
        args.out_dir,
        pool_path=args.pool,
        runs=args.runs,
        shapes=args.shapes,
        era5=args.era5,
        era5_chapter=args.era5_chapter,
    )


if __name__ == "__main__":
    cli()

"""Where each scorecard row's test fleet sits against its loaded extent. Read-only.

The loaded extent is the lon/lat range of the ERA5 grid a run loads, after its
own bbox slice (`CONTEXT.md`). A unit outside it has its winds extrapolated
from the edge of the grid rather than interpolated between cells.

This exists because the check of 2026-09-11 asked the wrong question. It
compared the European rows against the extent of the ERA5 files, which is what
the download covers, instead of against each row's bbox-sliced extent, which is
what a run actually loads. A row whose box stops inside the files passes the
first comparison and fails the second. DK is that case: its box stops at
13.5 degrees east, and Bornholm lies near 14.9. So the suspension notice of
2026-09-11 and the commit message of c480f46 both understated how many rows
carry extrapolated winds, and both are corrected in place.

For each scorecard row this loads the exact configuration behind it, the test
fleet run_evaluate scores, and the ERA5 the run would load, and reports the
loaded extent, the units outside it, their capacity share and how far out the
furthest lies. It opts in to extrapolation so that the audit itself never
refuses; nothing here is a run, and it writes no run directory.

The curve library differs by region and does not affect a coordinate, but the
fleets are loaded under the same input root each row was run on, so the audit
reads the same metadata the row did.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python scripts/analysis/extent_audit.py \
        <out_dir> [CODE ...]

With no codes it audits every row in ``baseline_bootstrap.CONFIGS``.
"""
from pathlib import Path

import pandas as pd

import baseline_bootstrap as bb
from vwf.harness.driver import load_obs_and_fleet
from vwf.cli.common import make_parser
from vwf.datasets.era5 import prep_era5
from vwf.harness.driver import era5_dir
from vwf.harness.regions import load_region
from vwf.wind import loaded_extent_coverage


def audit(code: str) -> dict:
    """One row's test fleet against the extent its configuration loads."""
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    year = int(spec.test_years[0])
    _, turb_info = load_obs_and_fleet(spec, year)
    # The extent is a property of the files and the box, not of a year or of
    # the roughness, so the wind fields are loaded without deriving z0.
    reanalysis = prep_era5(
        spec.code, train=False, calc_z0=False, bbox=spec.bbox,
        era5_dir=era5_dir(spec), allow_extrapolation=True,
    )
    record = loaded_extent_coverage(reanalysis, turb_info)
    lon_min, lon_max, lat_min, lat_max = record["loaded_extent"]
    return {
        "code": code,
        "config": bb.CONFIGS[code],
        "test_year": year,
        "bbox": ", ".join(str(v) for v in spec.bbox),
        "loaded_lon_min": lon_min, "loaded_lon_max": lon_max,
        "loaded_lat_min": lat_min, "loaded_lat_max": lat_max,
        "units": record["units"],
        "units_outside": record["units_outside_loaded_extent"],
        "capacity_share_outside": record["capacity_share_outside_loaded_extent"],
        "max_degrees_outside": record["max_degrees_outside_loaded_extent"],
    }


def main(out_dir: str, codes: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in codes or list(bb.CONFIGS):
        rows.append(audit(code))
        r = rows[-1]
        print(f"{code}: {r['units_outside']} of {r['units']} units outside, "
              f"{r['capacity_share_outside']:.4%} of capacity, "
              f"up to {r['max_degrees_outside']:.2f} degrees", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "extent_audit.csv", index=False)
    outside = frame.loc[frame["units_outside"] > 0, "code"].tolist()
    print(f"\n{len(outside)} of {len(frame)} rows audited carry extrapolated winds: "
          f"{', '.join(outside) if outside else 'none'}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir> [CODE ...]``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument("codes", nargs="*", metavar="CODE",
                        help="Scorecard rows to audit (default: every row in CONFIGS)")
    args = parser.parse_args(argv)
    main(args.out_dir, args.codes)


if __name__ == "__main__":
    cli()

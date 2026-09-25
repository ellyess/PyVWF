"""One configuration's cluster sweep, timed, before thirteen more are run.

`docs/findings/method-cluster-selection-prereg.md` registers a nested selection
protocol whose cost was estimated by extrapolation from a single anchor. This
measures the unit that cost is built from: one train run over the registered
cluster-count grid at one time slice, plus one evaluation, on the archive and
roughness treatment the rebuild uses.

**A country-level configuration has no cluster count to select.** The first
attempt asked Belgium for 2 clusters and
``pyvwf.data.assign_country_clusters`` refused: the country path runs no
clustering step, the grid points arrive with their cluster column already set,
and only 1 or that count is legal. The registered country-level grid of 1 to
100 cannot run, and the candidate set for each of the nine country rows has two
members. That is a finding about the registration, recorded here because the
script is what found it.

**It is not a fold of the registered protocol.** It trains on the
configuration's whole training window rather than on a fold of it, because the
fold structure is blocked: ``train_years`` is an inclusive ``[start, end]``
pair, validated as such in `pyvwf.harness.regions` and consumed as such by the
observation sources, so holding out a year from the middle of the window cannot
be expressed without changing that contract. The per-fit cost is the same
either way, which is what this measures, and nothing here is read as a result.

Read-only with respect to the tree apart from the run directories it writes
under ``<out_dir>``, which is under ``output/``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/studies/method-cluster-selection/cluster_sweep_cost.py <region-stem> <out_dir>
"""

import dataclasses
import time
from pathlib import Path

import pandas as pd

from pyvwf.cli.common import make_parser
from pyvwf.harness import driver, regions

#: The turbine-level grids the registration fixes.
ONSHORE_GRID = (1, 10, 25, 50, 100, 200, 500, 1000)
OFFSHORE_GRID = (1, 2, 3, 5, 10, 25, 50, 100)

#: Where a country-level configuration's cluster count comes from. It is not a
#: fitting choice: ``pyvwf.data.assign_country_clusters`` accepts 1 or the count
#: the grid points already carry and refuses everything else, because no
#: clustering step runs on the country path. So the candidate set has two
#: members and is read from the pool rather than declared.
POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")

#: Held constant by the registration.
ERA5_PATH, ROUGHNESS, TIME_SLICE = "era5/EU_2026-09", "derived", "fixed"


def grid_for(spec: regions.RegionSpec, mode: str, pool: Path = POOL) -> tuple[int, ...]:
    if spec.obs_level == "country":
        points = pd.read_csv(pool)
        present = int((points["country_code"] == spec.code).sum())
        return (1, present)
    return OFFSHORE_GRID if mode == "offshore" else ONSHORE_GRID


def main(stem: str, out_dir: str, mode: str = "all", pool: Path = POOL) -> None:
    out = Path(out_dir)
    spec = regions.load_region(Path("configs/regions") / f"{stem}.toml")
    grid = grid_for(spec, mode, pool)

    swept = dataclasses.replace(
        spec, cluster_list=grid, time_slices=(TIME_SLICE,), era5_path=ERA5_PATH, roughness=ROUGHNESS
    )
    print(
        f"{swept.code} {mode}: {len(grid)} cluster counts {grid}, "
        f"slice {TIME_SLICE}, archive {ERA5_PATH}, roughness {ROUGHNESS}",
        flush=True,
    )
    print(f"  train {swept.train_years}, test {swept.test_years}", flush=True)

    started = time.monotonic()
    train_dir = driver.run_train(swept, out, mode=mode, run_name="sweep-cost")
    trained = time.monotonic()
    evaluate_dir = driver.run_evaluate(swept, train_dir, out, mode=mode, run_name="sweep-cost")
    finished = time.monotonic()

    metrics = pd.read_csv(evaluate_dir / "metrics.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        columns = [
            c
            for c in (
                "variant",
                "num_clu",
                "time_res",
                "MBE",
                "MAE",
                "RMSE",
                "r",
                "Units",
                "Samples",
                "fit_quality",
                "substituted_capacity_share",
                "extrapolated_capacity_share",
                "excluded_share",
            )
            if c in metrics.columns
        ]
        print("\n=== every row of metrics.csv")
        print(metrics[columns].to_string(index=False))

    fitted = len(grid)
    print(f"\n=== cost, {swept.code} {mode}")
    print(
        f"  train      {trained - started:8.1f} s for {fitted} cluster counts "
        f"({(trained - started) / fitted:.1f} s each)"
    )
    print(f"  evaluate   {finished - trained:8.1f} s")
    print(f"  total      {finished - started:8.1f} s")
    print(f"\ntrain: {train_dir}\nevaluate: {evaluate_dir}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<region-stem> <out_dir> [mode]``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("stem", metavar="region-stem", help="A maintained config, e.g. be")
    parser.add_argument("out_dir", help="Directory for the runs, under output/")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=("all", "onshore", "offshore"),
        help="Fleet mode (default: all)",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=POOL,
        help=f"The control-point pool, for a country-level grid (default: {POOL})",
    )
    args = parser.parse_args(argv)
    main(args.stem, args.out_dir, args.mode, pool=args.pool)


if __name__ == "__main__":
    cli()

"""Is each national joint fit's target reachable? Classify every recorded period.

Registered in ``docs/findings/method-joint-fit-reachability-prereg.md``
(issue #68). For one country scorecard configuration, this trains the row
through the harness exactly as the scorecard does and, for every period the
joint national fit sees, records beside the fit's own outcome the range of
national capacity factor any offsets can reach.

The national capacity factor is a capacity-weighted sum of cluster capacity
factors, and each cluster's depends only on its own offset, so the reachable
range is separable: the weighted sum of each cluster's lowest and highest
capacity factor over offsets. Each extreme is taken on a 0.25 m/s grid over the
offsets the fit accepts (within ``OFFSET_XTOL`` of the ±10 m/s bounds), then
refined by a bounded one-dimensional search around the best grid point, with
the objective's own simulation (``train_simulate_wind_from_ws`` at the
period's fitted scalar) on winds interpolated once.

The fit is not changed: the wrapper computes the range, then calls the original
``find_offsets_country_level`` and records what ``minimize`` returned. The fit
outcomes are the control: they must reproduce the recorded licensed-curve run in
``output/country_curves_2026-09-25/diag/``.

Usage, from the repository root, one row per process:

    PYVWF_INPUT=input/combined PYTHONPATH=src python scripts/dev/run_locked.py -- \\
        python scripts/studies/method-joint-fit-reachability/reachability_pass.py \\
        <out_dir> <stem>
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import pyvwf.correction as correction
from pyvwf.cli.common import make_parser
from pyvwf.harness.driver import run_train
from pyvwf.harness.regions import load_region
from pyvwf.time_utils import parse_time_slice
from pyvwf.wind import interpolate_wind, train_simulate_wind_from_ws

REPO = Path(__file__).resolve().parents[3]
#: Grid step of the first pass over offsets, in m/s; each extreme is then refined.
GRID_STEP = 0.25
BOUND = 10.0


def cluster_extremes(ws, curves, scalar: float) -> tuple[float, float]:
    """(lowest, highest) capacity factor one cluster reaches over accepted offsets.

    The fit refuses an offset within ``OFFSET_XTOL`` of a bound, so the range is
    taken inside that margin. A grid finds the neighbourhood of each extreme,
    because a power curve with cut-out is not monotonic in the offset, and a
    bounded search refines it, so the range is not understated by the grid.
    """
    lo_edge = -BOUND + correction.OFFSET_XTOL
    hi_edge = BOUND - correction.OFFSET_XTOL
    grid = np.arange(lo_edge, hi_edge + 1e-12, GRID_STEP)
    grid[-1] = min(grid[-1], hi_edge)
    cf = np.array([float(train_simulate_wind_from_ws(ws, curves, scalar, o)) for o in grid])

    def refine(sign: float) -> float:
        best = int(np.nanargmax(sign * cf))
        left = max(lo_edge, grid[best] - GRID_STEP)
        right = min(hi_edge, grid[best] + GRID_STEP)
        res = minimize_scalar(
            lambda o: -sign * float(train_simulate_wind_from_ws(ws, curves, scalar, o)),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-5},
        )
        return sign * max(sign * cf[best], -res.fun)

    return refine(-1.0), refine(1.0)


def reachable_range(
    year, time_slice, scalars_by_cluster, turb_info, reanalysis, curves, seasons=None
) -> dict:
    """The national range, by cluster and in total, with the capacity weights."""
    months = parse_time_slice(time_slice, seasons)
    period = reanalysis.sel(
        time=np.logical_and(reanalysis.time.dt.year == year, reanalysis.time.dt.month.isin(months))
    )
    clusters = sorted(turb_info["cluster"].unique())
    capacity = turb_info.groupby("cluster")["capacity"].sum()
    total = float(capacity[clusters].sum())
    lo = hi = 0.0
    per_cluster = {}
    for c in clusters:
        members = turb_info[turb_info["cluster"] == c]
        ws = interpolate_wind(period, members)
        c_lo, c_hi = cluster_extremes(ws, curves, scalars_by_cluster.get(c, 1.0))
        weight = float(capacity[c]) / total
        lo += weight * c_lo
        hi += weight * c_hi
        per_cluster[str(c)] = {"weight": weight, "lo": c_lo, "hi": c_hi}
    return {"lo": lo, "hi": hi, "per_cluster": per_cluster}


def main(out_dir: str, stem: str) -> None:
    spec = load_region(REPO / "configs" / "regions" / "scorecard" / f"{stem}_country.toml")
    records: list[dict] = []
    current: dict = {}
    original_fit = correction.find_offsets_country_level
    original_minimize = correction.minimize

    def fit(*args, **kwargs):
        rng = reachable_range(
            kwargs["year"],
            kwargs["time_slice"],
            kwargs["scalars_by_cluster"],
            kwargs["turb_info"],
            kwargs["reanalysis"],
            kwargs["powerCurveFile"],
            kwargs.get("seasons"),
        )
        current.clear()
        current.update(
            year=int(kwargs["year"]),
            time_slice=str(kwargs["time_slice"]),
            n_clusters=int(kwargs["turb_info"]["cluster"].nunique()),
            obs=float(kwargs["obs_country_cf"]),
            **rng,
        )
        offsets = original_fit(*args, **kwargs)
        current["refused"] = bool(all(np.isnan(v) for v in offsets.values()))
        records.append(dict(current))
        return offsets

    def spy(fun, x0, *args, **kwargs):
        res = original_minimize(fun, x0, *args, **kwargs)
        x = np.asarray(res.x, dtype=float)
        lo, hi = zip(*kwargs["bounds"])
        at_bound = np.isclose(x, lo, atol=correction.OFFSET_XTOL) | np.isclose(
            x, hi, atol=correction.OFFSET_XTOL
        )
        current.update(
            success=bool(res.success),
            message=str(res.message),
            nit=int(res.nit),
            fun=float(res.fun),
            offsets=x.round(6).tolist(),
            n_at_bound=int(at_bound.sum()),
        )
        return res

    correction.find_offsets_country_level = fit
    correction.minimize = spy
    try:
        run_dir = run_train(spec, out_dir, run_name="reach")
    finally:
        correction.find_offsets_country_level = original_fit
        correction.minimize = original_minimize
    (run_dir / "reachability.json").write_text(json.dumps(records, indent=1) + "\n")
    print(f"{spec.code}: {len(records)} periods recorded in {run_dir / 'reachability.json'}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir> <stem>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the run directory, under output/")
    parser.add_argument("stem", help="Country stem, e.g. fr for fr_country.toml")
    args = parser.parse_args(argv)
    main(args.out_dir, args.stem)


if __name__ == "__main__":
    cli()

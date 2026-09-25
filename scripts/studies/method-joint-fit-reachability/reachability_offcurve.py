"""The reachability pass again, with off-curve corrected speeds counted as zero output.

A follow-up registered after the pass ran, in the addendum of
``docs/findings/method-joint-fit-reachability-prereg.md`` (issue #68). The pass
found every unreachable refusal below its range, with floors near 0.24 in Italy
even at a -10 m/s offset. The simulation drops a corrected speed off the curve
(below 0 m/s or above its 40 m/s end) from the mean rather than counting it as
zero, which keeps that floor high. This driver trains each row the same way and
records, for every period, the range with each cluster's corrected speeds
clipped to the curve's ends, where the curve reads zero, before the curve is
applied. Missing input speeds stay missing. The fit itself is not changed, and
its outcomes are the control again.

Usage, from the repository root, one row per process:

    PYVWF_INPUT=input/combined PYTHONPATH=src python scripts/dev/run_locked.py -- \\
        python scripts/studies/method-joint-fit-reachability/reachability_offcurve.py \\
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
GRID_STEP = 0.25
BOUND = 10.0
#: The curve table's speed range; outside it the interpolator returns no value.
CURVE_MIN, CURVE_MAX = 0.0, 40.0


def cf_zero(ws, curves, scalar: float, offset: float) -> float:
    """The cluster's capacity factor with off-curve corrected speeds counted as zero.

    The correction is applied here and the result clipped to the curve's ends,
    where the curve reads zero (below cut-in, above cut-out), then passed
    through the objective's simulation with the identity correction. A missing
    input speed stays missing through the clip.
    """
    corrected = (ws * scalar + offset).clip(CURVE_MIN, CURVE_MAX)
    return float(train_simulate_wind_from_ws(corrected, curves, 1.0, 0.0))


def off_curve_share(ws, scalar: float, offset: float) -> float:
    """Capacity-weighted share of present steps whose corrected speed is off the curve."""
    corrected = ws * scalar + offset
    present = corrected.notnull()
    off = present & ((corrected < CURVE_MIN) | (corrected > CURVE_MAX))
    weights = corrected["capacity"]
    return float((off * weights).sum() / (present * weights).sum())


def cluster_extremes_zero(ws, curves, scalar: float) -> dict:
    """(lowest, highest) CF with off-curve steps as zero, and where the lowest sits."""
    lo_edge = -BOUND + correction.OFFSET_XTOL
    hi_edge = BOUND - correction.OFFSET_XTOL
    grid = np.arange(lo_edge, hi_edge + 1e-12, GRID_STEP)
    grid[-1] = min(grid[-1], hi_edge)
    cf = np.array([cf_zero(ws, curves, scalar, o) for o in grid])

    def refine(sign: float) -> tuple[float, float]:
        best = int(np.nanargmax(sign * cf))
        left = max(lo_edge, grid[best] - GRID_STEP)
        right = min(hi_edge, grid[best] + GRID_STEP)
        res = minimize_scalar(
            lambda o: -sign * cf_zero(ws, curves, scalar, o),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-5},
        )
        if sign * cf[best] >= -res.fun:
            return float(cf[best]), float(grid[best])
        return float(sign * -res.fun), float(res.x)

    (lo, lo_at), (hi, _) = refine(-1.0), refine(1.0)
    return {
        "lo": lo,
        "hi": hi,
        "lo_offset": lo_at,
        "lo_off_curve": off_curve_share(ws, scalar, lo_at),
    }


def zero_range(year, time_slice, scalars_by_cluster, turb_info, reanalysis, curves, seasons=None):
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
        ws = interpolate_wind(period, turb_info[turb_info["cluster"] == c])
        ext = cluster_extremes_zero(ws, curves, scalars_by_cluster.get(c, 1.0))
        weight = float(capacity[c]) / total
        lo += weight * ext["lo"]
        hi += weight * ext["hi"]
        per_cluster[str(c)] = {"weight": weight, **ext}
    return {"lo_zero": lo, "hi_zero": hi, "per_cluster_zero": per_cluster}


def main(out_dir: str, stem: str) -> None:
    spec = load_region(REPO / "configs" / "regions" / "scorecard" / f"{stem}_country.toml")
    records: list[dict] = []
    current: dict = {}
    original_fit = correction.find_offsets_country_level
    original_minimize = correction.minimize

    def fit(*args, **kwargs):
        current.clear()
        current.update(
            year=int(kwargs["year"]),
            time_slice=str(kwargs["time_slice"]),
            n_clusters=int(kwargs["turb_info"]["cluster"].nunique()),
            obs=float(kwargs["obs_country_cf"]),
            **zero_range(
                kwargs["year"],
                kwargs["time_slice"],
                kwargs["scalars_by_cluster"],
                kwargs["turb_info"],
                kwargs["reanalysis"],
                kwargs["powerCurveFile"],
                kwargs.get("seasons"),
            ),
        )
        offsets = original_fit(*args, **kwargs)
        current["refused"] = bool(all(np.isnan(v) for v in offsets.values()))
        records.append(dict(current))
        return offsets

    def spy(fun, x0, *args, **kwargs):
        res = original_minimize(fun, x0, *args, **kwargs)
        current.update(
            success=bool(res.success),
            fun=float(res.fun),
            offsets=np.asarray(res.x, dtype=float).round(6).tolist(),
        )
        return res

    correction.find_offsets_country_level = fit
    correction.minimize = spy
    try:
        run_dir = run_train(spec, out_dir, run_name="offcurve")
    finally:
        correction.find_offsets_country_level = original_fit
        correction.minimize = original_minimize
    (run_dir / "reachability_offcurve.json").write_text(json.dumps(records, indent=1) + "\n")
    print(f"{spec.code}: {len(records)} periods recorded in {run_dir}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir> <stem>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the run directory, under output/")
    parser.add_argument("stem", help="Country stem, e.g. it for it_country.toml")
    args = parser.parse_args(argv)
    main(args.out_dir, args.stem)


if __name__ == "__main__":
    cli()

"""Is the pivot near 4 m/s physics, or where the offset tie-break defaults?

`method-correction-identifiability.md` reports that four dense rows' pencils of
correction lines cross at 3.81 to 4.06 m/s, independently. Four rows agreeing
at one speed is exactly the shape of a result that looks like physics and is
arithmetic, so this probes it before anything is built on it.

**The condition, fixed here before the probe runs.** The offset search
`vwf.correction.find_offset_iterative` starts at ``offset = 0`` with a step of
``sign(obs - sim) * 10.0`` m/s and halves whenever a proposed step exceeds the
last. If the fitted offset depends on that schedule, the pivot is where the
tie-break lands and is an artefact. **If re-solving from several different
initial steps returns the same offsets, the search is finding a root rather
than stopping somewhere, and the pivot is a property of the scalar rule and
the data, not of the initialisation.** Either answer is reported plainly.

Two measurements beside it:

- **Can the objective see below the pivot at all?** The fit matches one
  capacity-weighted mean capacity factor, so what the objective sees is energy.
  This reports the share of simulated capacity-factor mass contributed by days
  whose mean wind is below 4 m/s. A pivot in a region contributing almost
  nothing is an extrapolation of the fitted line, whatever else it is.
- **Does the pivot reproduce on a different fleet and archive?** The finding
  measured the chapter-era pool. This recomputes it on the selection study's
  own factors, which are a current fleet on `era5/EU_2026-09`.

Read-only with respect to the tree. Writes under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/studies/method-correction-identifiability/pivot_probe.py <out_dir> [DE|DK|UK]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vwf.correction import find_offset_iterative
from vwf.data import load_power_curves
from vwf.datasets.era5 import prep_era5
from vwf.harness import regions
from vwf.wind import fast_simulate_cf, interpolate_wind, prepare_offset_arrays

REPO = Path(__file__).resolve().parents[3]
SEL = REPO / "output/cluster_selection_2026-09-15"

#: The wind below which a turbine produces almost nothing, and where the pivot
#: was found.
CUT_IN = 4.0

#: Initial steps to re-solve from. 10.0 is the shipped default and 3.0 was the
#: default until 2026-02-14, so both are tested. 0.25 is below the natural step
#: size and is included to show where the schedule starts to throttle.
INITIAL_STEPS = (10.0, 4.0, 3.0, 1.0, 0.5, 0.25)

#: Iteration caps. 100 is shipped; 30 was the cap alongside the 3.0 default.
MAX_ITERS = (100, 30)

#: row label, config stem, fleet mode, the run's cluster count, its train dir.
ROWS = {
    "DE": ("de", "onshore", 500, "train-final"),
    "DK": ("dk", "onshore", 884, "train-onshore-final"),
    "UK": ("uk", "onshore", 300, "train-onshore-final"),
}
BBOX = {"dk": (7.5, 15.4, 54.0, 58.2)}


def pivot(a: np.ndarray, b: np.ndarray) -> float:
    va = float(np.var(a, ddof=1))
    return float("nan") if va <= 0 else float(-np.cov(a, b, ddof=1)[0, 1] / va)


def main(out_dir: str, *only: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    curves = load_power_curves()
    summary, per_cluster = [], []

    for label in (only or ROWS):
        stem, mode, k, train_dir = ROWS[label]
        spec = regions.load_region(REPO / "configs" / "regions" / f"{stem}.toml")
        base = SEL / label / train_dir
        factors = pd.read_csv(base / f"factors_fixed_{k}.csv")
        fleet = pd.read_csv(base / f"train_turb_info_{k}.csv")
        print(f"\n=== {label}: {len(fleet)} units, {len(factors)} clusters, "
              f"train {spec.train_years}", flush=True)

        reanalysis = prep_era5(spec.code, True, True,
                               bbox=BBOX.get(stem, spec.bbox),
                               era5_dir=REPO / "input" / "era5" / "EU_2026-09",
                               roughness="derived")
        first, last = spec.train_years
        years = pd.DatetimeIndex(reanalysis.time.values).year
        reanalysis = reanalysis.isel(time=np.where((years >= first) & (years <= last))[0])
        speed = interpolate_wind(reanalysis, fleet)
        speed = speed.transpose("time", "turbine")

        ids = pd.Index(np.asarray(speed["turbine"].values).astype(str))
        info = fleet.assign(ID=fleet["ID"].astype(str)).set_index("ID").loc[ids]
        cluster_of = info["cluster"].to_numpy()

        # What the objective can see: the share of simulated capacity-factor
        # mass from days below the pivot, on the uncorrected winds.
        whole = prepare_offset_arrays(speed, curves)
        cf = np.empty_like(whole["ws_data"])
        for akima, mask in whole["model_groups"]:
            cf[:, mask] = akima(whole["ws_data"][:, mask])
        cf = np.nan_to_num(cf)
        weights = np.broadcast_to(whole["capacities"][None, :], cf.shape)
        daily_mean_wind = np.nanmean(whole["ws_data"], axis=1)
        below = daily_mean_wind < CUT_IN
        share_days = float(below.mean())
        share_mass = float((cf[below] * weights[below]).sum()
                           / (cf * weights).sum())

        started = time.monotonic()
        rows = []
        for _, f in factors.iterrows():
            cl = f["cluster"]
            mask = cluster_of == cl
            if not mask.any():
                continue
            arrays = prepare_offset_arrays(speed.isel(turbine=np.where(mask)[0]), curves)
            target = fast_simulate_cf(arrays, float(f["scalar"]), float(f["offset"]))
            row = {"cluster": cl, "scalar": float(f["scalar"]),
                   "offset_shipped": float(f["offset"])}
            uncorrected = fast_simulate_cf(arrays, 1.0, 0.0)
            for step in INITIAL_STEPS:
                for cap in MAX_ITERS:
                    # sign(obs - sim) is zero at the shipped optimum, so the
                    # search is started off it by using the uncorrected mean as
                    # sim, which is what the real fit had.
                    probe = pd.Series({"obs": target, "sim": uncorrected,
                                       "scalar": float(f["scalar"])})
                    key = (f"offset_from_{step:g}" if cap == 100
                           else f"offset_from_{step:g}_iter{cap}")
                    row[key] = find_offset_iterative(
                        probe, arrays, max_iter=cap, initial_step=step)
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame["row"] = label
        per_cluster.append(frame)
        elapsed = (time.monotonic() - started) / 60

        variants = [c for c in frame.columns if c.startswith("offset_from_")]
        piv = {"shipped": pivot(frame["scalar"].to_numpy(),
                                frame["offset_shipped"].to_numpy())}
        for col in variants:
            piv[col.replace("offset_from_", "")] = pivot(
                frame["scalar"].to_numpy(), frame[col].to_numpy())
        worst = max(float(np.nanmax(np.abs(frame[c] - frame["offset_shipped"])))
                    for c in variants)
        failed = int(sum(frame[c].isna().sum() for c in variants))
        summary.append({"row": label, "clusters": len(frame),
                        "days_below_cut_in": round(share_days, 4),
                        "cf_mass_below_cut_in": round(share_mass, 5),
                        **{k2: round(v, 3) for k2, v in piv.items()},
                        "worst_offset_change": round(worst, 5),
                        "non_convergences": failed,
                        "minutes": round(elapsed, 1)})
        print("  pivot: " + ", ".join(f"{k2} {v:.3f}" for k2, v in piv.items()),
              flush=True)
        print(f"  worst offset change across initialisations: {worst:.5f}; "
              f"non-convergences {failed}", flush=True)

    pd.concat(per_cluster, ignore_index=True).to_csv(out / "pivot_per_cluster.csv",
                                                     index=False)
    frame = pd.DataFrame(summary)
    frame.to_csv(out / "pivot_probe.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== summary")
        print(frame.to_string(index=False))
    print("\nIf the pivot is unchanged across initialisations, the offset search "
          "found a root and the pivot is not an artefact of where it started.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], *sys.argv[2:])

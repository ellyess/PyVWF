"""Where a scorecard row's corrected capacity factors are missing, and why. Read-only.

A corrected value is missing (NaN) when the harness cannot simulate it, and
before 2026-09-11 a missing value silently dropped out of the score. This
recomputes the corrected wind speeds for each variant of one scorecard row,
through the harness's own ``apply`` and the same ERA5 input the evaluation
used. It checks that the recomputed capacity factors equal the saved
``cor_cf_*.csv`` frame, NaN for NaN. It then classifies every missing
unit-timestep (a unit-day: every scorecard row runs on daily ERA5) by route:

- ``input``: the uncorrected value is missing too, so nothing about the
  correction caused it;
- ``failed_factor``: the unit's cluster has no scalar or no offset for that
  time slice (a failed offset fit);
- ``above_curve``: the corrected speed is above the power curve table's last
  speed (40 m/s). The Akima interpolator returns NaN there, not zero output;
- ``below_curve``: the corrected speed is below 0 m/s (a negative offset larger
  than scalar times speed). Also NaN, not zero output;
- ``unexplained``: none of the above. The script stops if any occur.

A monthly value is the mean of the days present, so a unit-month can lose some
days and still be scored. Both effects are reported:

- a unit-month with every day missing, which the common-row scoring now
  excludes;
- a unit-month with some days missing, which is still scored, on the days
  that remain.

For country-level rows the unit is a grid point, and the national value at each
day is reweighted over the grid points present. So a missing grid-point day
never shows as a missing national month.

Fixed before any row was audited, on 2026-09-11: the routes above and their
order of precedence (``input`` first, then ``failed_factor``, then the two
curve routes); the reproduction check; and every variant of every one of the 17
scorecard rows. Nothing has been changed after results.

It needs ERA5, the local input root and the git-ignored run tree, so a third
party cannot run it. Usage, from the repository root, one region per process,
with ``PYVWF_INPUT`` as in the row's manifest:

    PYTHONPATH=src python scripts/studies/scorecard/missing_value_audit.py <CODE> <out_dir>

Outputs in ``<out_dir>``: ``<CODE>_missing_summary.csv`` (per variant),
``<CODE>_missing_by_cluster.csv`` and ``<CODE>_missing_by_unit.csv``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))  # the shared tools
import baseline_bootstrap as bb
from vwf.clustering import cluster_turbines
from vwf.data import assign_country_clusters, load_power_curves, val_set
from vwf.harness.corrections import fit_quality, get_correction
from vwf.harness.driver import era5_dir, resolve_source
from vwf.harness.regions import load_region
from vwf.wind import add_time_resolution_columns

ROUTES = ("input", "failed_factor", "above_curve", "below_curve", "unexplained")


def main(code, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    ev = next((bb.BACKFILL / code).glob("evaluate-*-backfill"))
    manifest = json.loads((ev / "run_manifest.json").read_text())
    year = int(manifest["evaluation_year"])
    train_dir = Path(manifest["trained_from"])
    is_country = spec.obs_level == "country"

    _, turb_info, reanalysis, power_curves = val_set(
        spec.code, True, "all", year_test=year, obs_level=spec.obs_level,
        source=resolve_source(spec, "test"), era5_dir=era5_dir(spec), bbox=spec.bbox,
    )
    top_speed = float(load_power_curves().iloc[:, 0].max())
    model = get_correction(spec.correction_model)
    unc = pd.read_csv(ev / "unc_cf.csv", parse_dates=["time"]).set_index("time")
    unc.columns = unc.columns.astype(str)

    summaries, by_cluster, by_unit = [], [], []
    for path in sorted(ev.glob("cor_cf_*.csv")):
        time_res, num_clu = path.stem.removeprefix("cor_cf_").rsplit("_", 1)
        label = f"{time_res}_{num_clu}"
        factors = pd.read_csv(train_dir / f"factors_{time_res}_{num_clu}.csv")
        if is_country:
            clus_info = assign_country_clusters(turb_info, int(num_clu))
        else:
            train_fleet = pd.read_csv(train_dir / f"train_turb_info_{num_clu}.csv")
            clus_info = cluster_turbines(int(num_clu), train_fleet, False, turb_info,
                                         min_cluster_size=spec.min_cluster_size)
        cor_ws, cor_cf = model.apply(reanalysis, clus_info, power_curves, factors, time_res,
                                     seasons=spec.seasons)
        cor_ws = cor_ws.set_index("time")
        cor_cf = cor_cf.set_index("time")
        cor_ws.columns = cor_ws.columns.astype(str)
        cor_cf.columns = cor_cf.columns.astype(str)

        saved = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        saved.columns = saved.columns.astype(str)
        cor_cf = cor_cf.loc[saved.index, saved.columns]
        cor_ws = cor_ws.loc[saved.index, saved.columns]
        same_nan = np.array_equal(np.isnan(cor_cf.to_numpy()), np.isnan(saved.to_numpy()))
        same_val = np.allclose(cor_cf.to_numpy(), saved.to_numpy(), equal_nan=True, rtol=0, atol=1e-12)
        if not (same_nan and same_val):
            raise SystemExit(f"{code} {label}: recomputed corrected CF differs from {path.name}")

        # Each unit's factors for each timestep, as correct_wind_speed looks them up.
        units = saved.columns
        cluster_of = clus_info.assign(ID=clus_info["ID"].astype(str)).set_index("ID")["cluster"]
        slices = add_time_resolution_columns(
            pd.DataFrame({"time": saved.index, "year": saved.index.year, "month": saved.index.month}),
            spec.seasons,
        )[time_res].to_numpy()
        lookup = factors.set_index(["cluster", time_res])
        factor_ok = np.ones(saved.shape, dtype=bool)
        for j, unit in enumerate(units):
            keys = list(zip([cluster_of[unit]] * len(slices), slices))
            row = lookup.reindex(keys)
            factor_ok[:, j] = row[["scalar", "offset"]].notna().all(axis=1).to_numpy()

        missing = np.isnan(saved.to_numpy())
        ws = cor_ws.to_numpy()
        route = np.full(saved.shape, "", dtype=object)
        unc_nan = np.isnan(unc.loc[saved.index, units].to_numpy())
        route[missing & unc_nan] = "input"
        rest = missing & ~unc_nan
        route[rest & ~factor_ok] = "failed_factor"
        rest &= factor_ok
        route[rest & (ws > top_speed)] = "above_curve"
        route[rest & (ws < 0)] = "below_curve"
        route[missing & (route == "")] = "unexplained"
        if (route == "unexplained").any():
            raise SystemExit(f"{code} {label}: {(route == 'unexplained').sum()} unexplained NaN")

        capacity = clus_info.assign(ID=clus_info["ID"].astype(str)).set_index("ID")["capacity"][units]
        ym = saved.index.to_period("M")
        steps = pd.DataFrame(missing, index=ym, columns=units)
        month_missing = steps.groupby(level=0).mean()
        wholly = (month_missing == 1.0)
        partly = (month_missing > 0) & (month_missing < 1.0)

        counts = {r: int((route == r).sum()) for r in ROUTES}
        units_hit = {r: [u for j, u in enumerate(units) if (route[:, j] == r).any()] for r in ROUTES}
        cap_total = float(capacity.sum())
        quality = fit_quality(factors)
        summaries.append({
            "region": code, "variant": label, "reported": label == bb.REPORTED[code],
            "level": spec.obs_level, "units": len(units), "unit_steps": int(missing.size),
            **{f"steps_{r}": counts[r] for r in ROUTES},
            "missing_share": float(missing.mean()),
            **{f"units_{r}": len(units_hit[r]) for r in ROUTES},
            **{f"capacity_share_{r}": float(capacity[units_hit[r]].sum()) / cap_total for r in ROUTES},
            "unit_months_wholly_missing": int(wholly.to_numpy().sum()),
            "unit_months_partly_missing": int(partly.to_numpy().sum()),
            "partly_missing_mean_step_share": float(month_missing[partly].stack().mean())
            if partly.to_numpy().any() else 0.0,
            "max_scalar": quality["max_scalar"], "n_implausible_scalar": quality["n_implausible_scalar"],
            "n_failed_offset": quality["n_failed_offset"],
            "degenerate_clusters": quality["degenerate_clusters"],
        })

        clusters = cluster_of[units].to_numpy()
        for cl in sorted(set(clusters)):
            cols = clusters == cl
            f = factors[factors["cluster"] == cl]
            by_cluster.append({
                "region": code, "variant": label, "cluster": cl, "units": int(cols.sum()),
                "capacity_share": float(capacity[cols].sum()) / cap_total,
                "scalar_min": float(f["scalar"].min()), "scalar_max": float(f["scalar"].max()),
                "offset_min": float(f["offset"].min()), "offset_max": float(f["offset"].max()),
                **{f"steps_{r}": int((route[:, cols] == r).sum()) for r in ROUTES},
                "missing_share": float(missing[:, cols].mean()),
            })
        for j, unit in enumerate(units):
            if missing[:, j].any():
                by_unit.append({
                    "region": code, "variant": label, "ID": unit, "cluster": clusters[j],
                    "capacity": float(capacity[unit]),
                    **{f"steps_{r}": int((route[:, j] == r).sum()) for r in ROUTES},
                    "months_wholly_missing": int(wholly[unit].sum()),
                    "months_partly_missing": int(partly[unit].sum()),
                })
        print(f"{code} {label}: {missing.mean():.2%} of unit-steps missing "
              + ", ".join(f"{r} {counts[r]}" for r in ROUTES if counts[r]))

    pd.DataFrame(summaries).to_csv(out_dir / f"{code}_missing_summary.csv", index=False)
    pd.DataFrame(by_cluster).to_csv(out_dir / f"{code}_missing_by_cluster.csv", index=False)
    pd.DataFrame(by_unit).to_csv(out_dir / f"{code}_missing_by_unit.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

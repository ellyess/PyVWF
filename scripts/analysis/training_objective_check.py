"""Does a country-level fit's objective drop the days its own factors push off the curve? Read-only.

The country-level offset fit (``correction.find_offsets_country_level``)
minimises, for each training year and time slice, the squared difference
between the observed national CF and a capacity-weighted mean of the clusters'
simulated CF. Each cluster's CF is ``train_simulate_wind``'s capacity-weighted
mean over grid points and days, and that mean skips NaN. A corrected speed
below 0 m/s or above the curve table's 40 m/s gives NaN, so a day the factors
push off the curve drops out of the objective. It does not count as zero
output.

For one country-level scorecard row and configuration, this rebuilds the
training winds through ``train_set``, applies the fitted factors from the
row's training directory, and reports, per cluster and training year:

- the grid-point-days whose uncorrected speed is already off the curve;
- the grid-point-days the corrected speed puts below 0 or above 40 m/s;
- the cluster's simulated CF as the objective computes it (off-curve days
  skipped), and with those days counted as zero output;
- the national CF both ways, against the observed national CF for the period.
  The observation comes from ``cluster_train_set``, as the fit uses it.

It applies the saved factors, which ``format_bc_factors`` has reduced over the
training years, not the per-year offsets the optimiser found. So the national
CF here is the fitted correction's, not the optimiser's exact value in any one
year.

Fixed before it was run, on 2026-09-11: ES ``fixed_4`` (the scorecard row, whose
clusters 0 and 3 carry offsets of -5.64 and -4.46 m/s at scalars near 0.5), and
the quantities above. It needs ERA5 and the local input root, so a third party
cannot run it.

Usage, from the repository root:

    PYTHONPATH=src python scripts/analysis/training_objective_check.py <CODE> <time_res> <k> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_bootstrap as bb
from vwf.data import cluster_train_set, load_power_curves, train_set
from vwf.harness import driver
from vwf.harness.regions import load_region
from vwf.wind import _get_power_curve_cache, interpolate_wind


def main(code, time_res, k, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    train_dir = Path(f"output/validation/refresh_2026-08-24/{code}/train-refresh")
    if time_res != "fixed":
        raise SystemExit("only the fixed slice is implemented: one objective per training year")
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        spec.code, True, "all", obs_level=spec.obs_level,
        source=driver.resolve_source(spec), era5_dir=driver._era5_dir(spec), bbox=spec.bbox,
    )
    fitted = pd.read_csv(train_dir / f"train_turb_info_{k}.csv")
    fitted["ID"] = fitted["ID"].astype(str)
    turb_info = turb_info.assign(ID=turb_info["ID"].astype(str)).drop(columns="cluster", errors="ignore")
    turb_info = turb_info.merge(fitted[["ID", "cluster"]], on="ID", how="inner")
    factors = pd.read_csv(train_dir / f"factors_{time_res}_{k}.csv").set_index("cluster")
    bias, _ = cluster_train_set(gen_cf, time_res, int(k), turb_info, obs_level="country")
    obs_by_year = bias.groupby("year")["obs"].first()

    top = float(load_power_curves().iloc[:, 0].max())
    _, curves = _get_power_curve_cache(power_curves)
    ws = interpolate_wind(reanalysis, turb_info)            # (time, turbine)
    ws = ws.transpose("time", "turbine")
    years = pd.DatetimeIndex(ws["time"].values).year
    model = np.asarray(ws["model"].values)
    capacity = np.asarray(ws["capacity"].values, float)
    cluster = turb_info.set_index("ID").loc[np.asarray(ws["turbine"].values, dtype=str), "cluster"].to_numpy()
    unc = ws.to_numpy()

    rows, national = [], []
    for year in sorted(set(years)):
        t = years == year
        skipped_sum, skipped_w, zero_sum, zero_w = {}, {}, {}, {}
        for cl in sorted(set(cluster)):
            j = cluster == cl
            scalar, offset = factors.loc[cl, "scalar"], factors.loc[cl, "offset"]
            u = unc[np.ix_(t, j)]
            c = u * scalar + offset
            cf = np.full(c.shape, np.nan)
            for m in set(model[j]):
                cols = model[j] == m
                cf[:, cols] = curves[m](c[:, cols])
            w = np.broadcast_to(capacity[j][None, :], c.shape)
            present = ~np.isnan(cf)
            unc_off = (u < 0) | (u > top)
            below, above = (c < 0), (c > top)
            skipped_sum[cl] = float((cf[present] * w[present]).sum())
            skipped_w[cl] = float(w[present].sum())
            filled = np.where(below | above, 0.0, cf)
            ok = ~np.isnan(filled)
            zero_sum[cl] = float((filled[ok] * w[ok]).sum())
            zero_w[cl] = float(w[ok].sum())
            rows.append({
                "region": code, "config": f"{time_res}_{k}", "year": int(year), "cluster": int(cl),
                "scalar": scalar, "offset": offset, "grid_points": int(j.sum()),
                "grid_point_days": int(c.size), "uncorrected_off_curve": int(unc_off.sum()),
                "corrected_below_0": int(below.sum()), "corrected_above_curve": int(above.sum()),
                "weight_share_dropped": 1 - skipped_w[cl] / float(w.sum()),
                "cluster_cf_objective": skipped_sum[cl] / skipped_w[cl] if skipped_w[cl] else np.nan,
                "cluster_cf_zero_fill": zero_sum[cl] / zero_w[cl] if zero_w[cl] else np.nan,
            })
        cap_cl = pd.Series(capacity).groupby(cluster).sum()
        obj = sum(cap_cl[c] * skipped_sum[c] / skipped_w[c] for c in cap_cl.index if skipped_w[c]) / cap_cl.sum()
        zf = sum(cap_cl[c] * zero_sum[c] / zero_w[c] for c in cap_cl.index if zero_w[c]) / cap_cl.sum()
        national.append({"region": code, "config": f"{time_res}_{k}", "year": int(year),
                         "observed": float(obs_by_year.get(year, np.nan)),
                         "national_cf_objective": obj, "national_cf_zero_fill": zf})

    by_cluster = pd.DataFrame(rows)
    nat = pd.DataFrame(national)
    by_cluster.to_csv(out_dir / f"{code}_{time_res}_{k}_training_by_cluster.csv", index=False)
    nat.to_csv(out_dir / f"{code}_{time_res}_{k}_training_national.csv", index=False)
    with pd.option_context("display.width", 220):
        agg = by_cluster.groupby("cluster").agg(
            scalar=("scalar", "first"), offset=("offset", "first"), days=("grid_point_days", "sum"),
            unc_off=("uncorrected_off_curve", "sum"), below_0=("corrected_below_0", "sum"),
            above=("corrected_above_curve", "sum"), dropped_share=("weight_share_dropped", "mean"),
            cf_objective=("cluster_cf_objective", "mean"), cf_zero_fill=("cluster_cf_zero_fill", "mean"))
        print(agg.round(4).to_string())
        print(nat.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:5])

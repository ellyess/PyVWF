"""Paired bootstrap intervals for a scorecard row, as it was reported. Read-only.

For one scorecard row, this rebuilds the paired simulated and observed series
from the row's evaluate frames (``unc_cf.csv``, ``cor_cf_*.csv`` under
``output/validation/curve_resolution_backfill_2026-09-11/``). It reads the
observations through the code path ``run_evaluate`` uses, without ERA5. It then
resamples, and reports 95% percentile intervals for:

- uncorrected RMSE;
- corrected RMSE, for the row's reported configuration;
- the correction gain, uncorrected minus corrected RMSE;
- the paired difference between each other corrected configuration in the same
  run and the reported one. These are real near-equal conditions, so their
  widths show what a paired comparison of two similar conditions can resolve.

It was written to size the curve library study's baseline conditions (C0 for
the country-level rows, T0 for the turbine-level rows) before its gates were
fixed. The same output is the evidence for the 2026-09-11 correction notice
in ``docs/findings/scorecard.md``.

Fixed before any interval was computed, on 2026-09-11:

- 1,000 draws, seed 20260911, 95% percentile intervals;
- the resampling unit: months for country-level rows (the 12 paired monthly
  means ``country_skill`` scores), and units for turbine-level rows, after
  the pseudo-replicate collapse;
- paired draws: the same indices for every condition of a row;
- a reproduction check: the rebuilt point metrics must equal the row's
  ``metrics.csv`` to 1e-12, or the script stops before resampling;
- the reported configuration per row, from the scorecard's Best cfg column.

Changed after results were seen, on the same day:

- **NO's reported configuration.** The first version took NO's lowest
  corrected RMSE (``season_4``, 0.0384), since the scorecard names no winning
  configuration for NO. The scorecard row shows ``fixed_4`` (RMSE 0.039, MBE
  -0.030), so NO was re-run on ``fixed_4``. Its gain interval moved from
  -0.025 to 0.014 to -0.026 to 0.017. NO is excluded from the study's G1
  either way, because its correction gain is negative.
- **Rows added.** The first run covered the twelve rows the study uses (eight
  country-level, plus DE, DK, UK and US). BR, AU-NEM, NZ, CL and AR were added
  after the UK result, to check whether its pattern appears in any other row.
- **A unit with no complete row in one condition.** In CL, four plants have no
  corrected value, and the first version stopped on them. They now get weight
  0 in that condition, which is what ``skill_metrics`` does, so the rebuilt
  metrics still reproduce ``metrics.csv``. That means the intervals inherit the
  harness's scoring of each condition on its own rows. For CL (four plants),
  the US (10 of 6,078 rows) and AR's ``season_10`` (2 rows), the uncorrected
  and corrected figures are not on the same rows.

The intervals understate the uncertainty, for four reasons:

- they are conditional on each row's single test year;
- resampling 12 months independently ignores seasonality and serial
  correlation, and percentile intervals run narrow at n = 12;
- resampling units treats them as independent, although neighbouring units
  share weather;
- the country-level configurations were chosen on the test year, and the
  resampling leaves that choice out.

It reads the run tree under the git-ignored ``output/`` and the local input
tree, which holds confidential observations and the licensed curve library. So
a third party cannot run it. It is committed so that the cited intervals can
be regenerated from the runs that produced them.

Usage, from the repository root, one region per process, with ``PYVWF_INPUT``
set to the input root in the row's manifest (``input/combined`` for DE, DK,
UK, US, BR, AU-NEM and NZ; the default for the others):

    PYTHONPATH=src python scripts/analysis/baseline_bootstrap.py <CODE> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vwf.harness.bootstrap import (
    percentile_interval,
    resample_counts,
    resample_indices,
    rmse_over_rows,
    weighted_rmse,
)
from vwf.harness.driver import country_skill, tidy_eval_frame, load_obs_and_fleet
from vwf.harness.regions import load_region
from vwf.harness.skill import collapse_pseudo_replicates, skill_metrics

N_DRAWS = 1000
SEED = 20260911
BACKFILL = Path("output/validation/curve_resolution_backfill_2026-09-11")
CONFIGS = {
    "BE": "be_country", "ES": "es_country", "FR": "fr_country", "IE": "ie_country",
    "IT": "it_country", "NO": "no_country", "PT": "pt_country", "SE": "se_country",
    "DE": "de_k100", "DK": "dk_k100", "UK": "uk_k50", "US": "us_k250",
    "BR": "br_k60", "AU-NEM": "au_nem_k45", "NZ": "nz_k7", "CL": "cl_k10", "AR": "ar_k10",
}
# The scorecard's reported configuration per row, from its Best cfg column. NO
# names no winning configuration; its row shows fixed_4 (RMSE 0.039, MBE -0.030).
REPORTED = {
    "FR": "fixed_10", "BE": "season_3", "ES": "fixed_4", "IE": "season_1",
    "SE": "fixed_4", "IT": "season_3", "PT": "season_1", "NO": "fixed_4",
    "DE": "fixed_100", "DK": "season_100", "UK": "fixed_50", "US": "fixed_250",
    "BR": "fixed_60", "AU-NEM": "season_45", "NZ": "fixed_7", "CL": "fixed_10", "AR": "fixed_10",
}


def country_monthly(sim_cf, obs, turb_info):
    """The 12 paired monthly means country_skill scores, as arrays."""
    grid = [c for c in sim_cf.columns if c != "time"]
    cap = turb_info.assign(ID=turb_info["ID"].astype(str)).set_index("ID")["capacity"]
    valid = [c for c in grid if str(c) in cap.index]
    caps = cap[[str(c) for c in valid]].to_numpy(float)
    vals = sim_cf[valid].to_numpy(float)
    present = ~np.isnan(vals)
    wsum = np.where(present, caps, 0.0).sum(axis=1)
    country = np.where(wsum > 0, np.where(present, vals * caps, 0.0).sum(axis=1) / wsum, np.nan)
    sim = pd.DataFrame({"ym": pd.to_datetime(sim_cf["time"]).dt.to_period("M"), "cf_sim": country})
    sim_m = sim.groupby("ym")["cf_sim"].mean()
    o = obs.assign(ym=pd.to_datetime(obs["time"]).dt.to_period("M")).groupby("ym")["obs"].mean()
    both = pd.concat([sim_m, o.rename("cf_obs")], axis=1).dropna()
    return both


def main(code, out_dir):
    spec = load_region(Path("configs/regions/scorecard") / f"{CONFIGS[code]}.toml")
    ev = next((BACKFILL / code).glob("evaluate-*-backfill"))
    manifest = json.loads((ev / "run_manifest.json").read_text())
    year = int(manifest["evaluation_year"])
    metrics = pd.read_csv(ev / "metrics.csv")
    obs, turb_info = load_obs_and_fleet(spec, year)

    frames = {"uncorrected": pd.read_csv(ev / "unc_cf.csv")}
    for p in sorted(ev.glob("cor_cf_*.csv")):
        frames[p.stem.removeprefix("cor_cf_")] = pd.read_csv(p)

    # 1. Reproduce metrics.csv exactly, per variant.
    point, repro = {}, []
    for name, sim in frames.items():
        if name == "uncorrected":
            row = metrics[metrics.variant == "uncorrected"].iloc[0]
        else:
            ts, k = name.rsplit("_", 1)
            row = metrics[(metrics.time_res == ts) & (metrics.num_clu == int(k))
                          & (metrics.variant != "uncorrected")].iloc[0]
        if spec.obs_level == "country":
            got = country_skill(sim, obs, turb_info)
        else:
            got = skill_metrics(collapse_pseudo_replicates(tidy_eval_frame(sim, obs, turb_info), spec))
        for m in ("rmse", "mbe"):
            repro.append({"variant": name, "metric": m, "metrics_csv": row[m], "rebuilt": got[m],
                          "abs_diff": abs(row[m] - got[m])})
        point[name] = got
    repro = pd.DataFrame(repro)
    if repro.abs_diff.max() > 1e-12:
        raise SystemExit(f"{code}: rebuilt metrics do not reproduce metrics.csv\n{repro}")

    reported = REPORTED[code]

    rows = []
    if spec.obs_level == "country":
        series = {n: country_monthly(s, obs, turb_info) for n, s in frames.items()}
        n = len(series["uncorrected"])
        idx = resample_indices(n, seed=SEED, n_draws=N_DRAWS)

        def boot_rmse(name):
            d = (series[name].cf_sim - series[name].cf_obs).to_numpy()
            return rmse_over_rows(d, idx), np.sqrt((d ** 2).mean())
        n_resampled = n
    else:
        tidy = {nm: collapse_pseudo_replicates(tidy_eval_frame(s, obs, turb_info), spec)
                .dropna(subset=["cf_sim", "cf_obs", "capacity"]) for nm, s in frames.items()}
        units = np.array(sorted(tidy["uncorrected"].ID.unique()))
        n = len(units)
        counts = resample_counts(n, seed=SEED, n_draws=N_DRAWS)

        def boot_rmse(name):
            t = tidy[name].assign(w=lambda f: f.capacity, e=lambda f: f.capacity * (f.cf_sim - f.cf_obs) ** 2)
            # A unit with no complete row in this condition (all NaN, as in a
            # failed-offset cluster) is dropped by skill_metrics; weight 0 here
            # is the same thing. Units are those of the uncorrected frame.
            g = t.groupby("ID")[["w", "e"]].sum()
            extra = set(g.index) - set(units)
            if extra:
                raise SystemExit(f"{code} {name}: {len(extra)} units not in uncorrected")
            missing = len(units) - len(g)
            if missing:
                print(f"{code} {name}: {missing} unit(s) with no complete rows, weight 0")
            g = g.reindex(units, fill_value=0.0)
            w, e = g.w.to_numpy(), g.e.to_numpy()
            return weighted_rmse(counts, e, w), np.sqrt(e.sum() / w.sum())
        n_resampled = n

    boot = {nm: boot_rmse(nm) for nm in frames}
    for nm, (b, pt) in boot.items():
        if not np.isclose(pt, point[nm]["rmse"], rtol=0, atol=1e-12):
            raise SystemExit(f"{code} {nm}: bootstrap point estimate {pt} != {point[nm]['rmse']}")

    def add(quantity, draws_, est):
        lo, hi = percentile_interval(draws_)
        rows.append({"region": code, "level": spec.obs_level, "resampled": n_resampled,
                     "quantity": quantity, "estimate": est, "ci_lo": lo, "ci_hi": hi,
                     "width": hi - lo, "sd": float(np.std(draws_, ddof=1)),
                     "consistent_with_zero": bool(lo <= 0 <= hi) if "minus" in quantity else None})

    u, u_pt = boot["uncorrected"]
    c, c_pt = boot[reported]
    add("uncorrected RMSE", u, u_pt)
    add(f"corrected RMSE ({reported})", c, c_pt)
    add(f"gain: uncorrected minus corrected ({reported})", u - c, u_pt - c_pt)
    # Paired differences between two corrected configurations of the same run:
    # real near-equal conditions, so their interval shows what a paired
    # comparison of two similar conditions can resolve.
    for nm in frames:
        if nm in ("uncorrected", reported):
            continue
        b, pt = boot[nm]
        add(f"corrected RMSE {nm} minus {reported}", b - c, pt - c_pt)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"{code}_bootstrap.csv", index=False)
    repro.assign(region=code).to_csv(out_dir / f"{code}_reproduction.csv", index=False)
    print(f"{code}: reproduced {len(repro)} metric values (max abs diff {repro.abs_diff.max():.1e}); "
          f"reported {reported}; resampled {n_resampled}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

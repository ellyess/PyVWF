"""How concentrated a turbine-level scorecard row's error is across units. Read-only.

For one turbine-level row, reported configuration only, from the same evaluate
frames as ``baseline_bootstrap.py``:

- the capacity-effective number of units, (sum of w) squared over the sum of
  w squared, where w is a unit's summed capacity weight over the test year;
- the share of capacity-weighted squared error carried by the top 1, top 5 and
  top 1% of units, uncorrected and corrected;
- for the five units that carry the most corrected error: their capacity
  share, their error shares and their own RMSE, uncorrected and corrected.
  No unit identifiers are written, because some observations are confidential;
- leave one unit out: the range of the correction gain, in RMSE and in MAE,
  with each unit dropped in turn;
- resampled intervals for the RMSE gain and the MAE gain, from the same draws
  and seed as ``baseline_bootstrap.py``. The MAE interval shows whether
  fragility is specific to RMSE.

It is the evidence for the concentration figures in the 2026-09-11 correction
notice in ``docs/findings/scorecard.md``.

What was decided when:

- **Before it was written:** the UK's resampled gain interval included zero.
  An ad hoc check of the UK then gave the capacity-effective count (104 of
  348) and the top-1 and top-5 shares of corrected squared error (37% and
  60%). This script computes the same quantities, and reproduces those UK
  figures exactly.
- **In the first version, before any row but the UK was computed:** the top 1%
  share, leave-one-out, the MAE interval and the five-unit detail.
- **Changed after results.** In the US, one unit's corrected weight differs
  from its uncorrected weight (10 rows have no corrected value), and that
  tripped an assertion. Each condition now keeps its own weights, as
  ``metrics.csv`` does. The UK, DE and DK outputs were unchanged by this.
  CL's four plants with no corrected value then stopped the script. They now
  get weight 0 in the corrected condition, as in ``baseline_bootstrap.py``,
  and inherit the same caveat: the two conditions are not scored on the same
  rows for CL, the US and AR's ``season_10``.
- **Rows.** DE, DK and US were requested alongside the UK. BR, AU-NEM, NZ, CL
  and AR were added afterwards, to check the rows with few units.

It needs the same local run tree and input root as ``baseline_bootstrap.py``,
and is committed for the same reason: a third party cannot run it, but the
cited figures can be regenerated from the runs that produced them.

Usage, from the repository root, one region per process, with ``PYVWF_INPUT``
as in the row's manifest:

    PYTHONPATH=src python scripts/studies/scorecard/unit_concentration.py <CODE> <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))  # the shared tools
import baseline_bootstrap as bb
from vwf.cli.common import make_parser
from vwf.harness.driver import load_obs_and_fleet
from vwf.harness.bootstrap import percentile_interval, resample_counts, weighted_mean, weighted_rmse
from vwf.harness.driver import tidy_eval_frame
from vwf.harness.regions import load_region
from vwf.harness.skill import collapse_pseudo_replicates


def main(code, out_dir, backfill=bb.BACKFILL):
    spec = load_region(Path("configs/regions/scorecard") / f"{bb.CONFIGS[code]}.toml")
    ev = next((Path(backfill) / code).glob("evaluate-*-backfill"))
    year = int(json.loads((ev / "run_manifest.json").read_text())["evaluation_year"])
    obs, turb_info = load_obs_and_fleet(spec, year)
    names = {"unc": "unc_cf.csv", "cor": f"cor_cf_{bb.REPORTED[code]}.csv"}
    per_unit = {}
    for k, f in names.items():
        t = collapse_pseudo_replicates(tidy_eval_frame(pd.read_csv(ev / f), obs, turb_info), spec)
        t = t.dropna(subset=["cf_sim", "cf_obs", "capacity"])
        d = t.cf_sim - t.cf_obs
        t = t.assign(w=t.capacity, e=t.capacity * d**2, a=t.capacity * d.abs())
        per_unit[k] = t.groupby("ID")[["w", "e", "a"]].sum()
    units = per_unit["unc"].index
    # A unit with no complete corrected row gets weight 0 there, as skill_metrics drops it.
    assert set(per_unit["cor"].index) <= set(units)
    out_missing = len(units) - len(per_unit["cor"])
    u, c = per_unit["unc"], per_unit["cor"].reindex(units, fill_value=0.0)
    # Weights can differ where a corrected unit-month is NaN (US); each
    # condition keeps its own, as metrics.csv does.
    out_mismatch = int((~np.isclose(u.w, c.w)).sum())
    w, wc = u.w.to_numpy(), c.w.to_numpy()
    n = len(units)

    def rmse(e, ww):
        return np.sqrt(e.sum() / ww.sum())

    out = {
        "region": code,
        "units": n,
        "units_with_weight_mismatch": out_mismatch,
        "units_missing_when_corrected": out_missing,
        "capacity_effective_n": float(w.sum() ** 2 / (w**2).sum()),
    }
    top1pct = max(1, int(np.ceil(0.01 * n)))
    out["top1pct_n_units"] = top1pct
    for k, g in (("unc", u), ("cor", c)):
        s = (g.e / g.e.sum()).sort_values(ascending=False)
        out[f"{k}_sse_top1"] = float(s.iloc[0])
        out[f"{k}_sse_top5"] = float(s.head(5).sum())
        out[f"{k}_sse_top1pct"] = float(s.head(top1pct).sum())
    out["rmse_gain"] = float(rmse(u.e, u.w) - rmse(c.e, c.w))
    out["mae_gain"] = float(u.a.sum() / w.sum() - c.a.sum() / wc.sum())

    # Leave one unit out.
    E_u, E_c, A_u, A_c, W, Wc = u.e.sum(), c.e.sum(), u.a.sum(), c.a.sum(), w.sum(), wc.sum()
    loo_rmse = np.sqrt((E_u - u.e.values) / (W - w)) - np.sqrt((E_c - c.e.values) / (Wc - wc))
    loo_mae = (A_u - u.a.values) / (W - w) - (A_c - c.a.values) / (Wc - wc)
    out["loo_rmse_gain_min"], out["loo_rmse_gain_max"] = (
        float(loo_rmse.min()),
        float(loo_rmse.max()),
    )
    out["loo_mae_gain_min"], out["loo_mae_gain_max"] = float(loo_mae.min()), float(loo_mae.max())
    top_c = (c.e / c.e.sum()).sort_values(ascending=False).index[0]
    out["gain_rmse_without_top_cor_unit"] = float(loo_rmse[list(units).index(top_c)])

    # Same draws as baseline_bootstrap.py: units sorted, same seed, same order.
    order = np.argsort(np.array(units.astype(str)))
    counts = resample_counts(n, seed=bb.SEED, n_draws=bb.N_DRAWS)
    uu, cc = u.iloc[order], c.iloc[order]
    g_rmse = weighted_rmse(counts, uu.e.values, uu.w.values) - weighted_rmse(
        counts, cc.e.values, cc.w.values
    )
    g_mae = weighted_mean(counts, uu.a.values, uu.w.values) - weighted_mean(
        counts, cc.a.values, cc.w.values
    )
    out["rmse_gain_ci_lo"], out["rmse_gain_ci_hi"] = percentile_interval(g_rmse)
    out["mae_gain_ci_lo"], out["mae_gain_ci_hi"] = percentile_interval(g_mae)

    top5 = (c.e / c.e.sum()).sort_values(ascending=False).head(5).index
    detail = pd.DataFrame(
        {
            "rank": range(1, 6),
            "capacity_weight_share": (u.w[top5] / W).values,
            "unc_sse_share": (u.e[top5] / E_u).values,
            "cor_sse_share": (c.e[top5] / E_c).values,
            "unit_rmse_unc": np.sqrt(u.e[top5] / u.w[top5]).values,
            "unit_rmse_cor": np.sqrt(c.e[top5] / c.w[top5]).values,
        }
    )
    out_dir = Path(out_dir)
    pd.DataFrame([out]).to_csv(out_dir / f"{code}_concentration.csv", index=False)
    detail.to_csv(out_dir / f"{code}_top5_units.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(pd.Series(out).to_string())
        print(detail.to_string(index=False))


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<CODE> <out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("code", help="Scorecard row, a key of CONFIGS, e.g. DK")
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument(
        "--backfill",
        type=Path,
        default=bb.BACKFILL,
        help=f"The rows' evaluate runs (default: {bb.BACKFILL})",
    )
    args = parser.parse_args(argv)
    main(args.code, args.out_dir, backfill=args.backfill)


if __name__ == "__main__":
    cli()

"""Does the affine correction survive at hourly resolution? (Chile, 2024)

Every validated result in this repo scores MONTHLY capacity factors. Any use of
the correction inside a forecasting system would score it hourly, so this asks
whether the correction is doing something real at the timescale forecasting
cares about, or whether it only works on aggregates.

WHY CHILE, not Brazil (the region was chosen on a hard constraint, before any
result was seen). The test needs the correction to have been FITTED on hourly
winds, otherwise a failure at hourly could just be a daily-to-hourly wind
distribution mismatch acting through the convex power curve. Checking the
configs: Brazil, the US and AU-NEM point at daily-mean ERA5 (era5/BR_daily,
era5/US_daily); only Chile and New Zealand use hourly ERA5. Of those two, Chile
has 59 plants against New Zealand's 12, and Chile's CEN timestamps are a fixed
UTC-4 with no daylight saving, whereas New Zealand's EMI data is in trading
periods with 50-period DST days. Chile is therefore the clean case.

Chile carries its own scientific caveat (ERA5 exaggerates the north-south wind
gradient, so the correction removes the mean bias but adds limited skill). That
caveat is about SPATIAL bias and is separate from the question here, which is
about TIMESCALE. It does mean the monthly gain being tested against is modest.

Design: take the correction already trained and validated for Chile (affine,
k=10, fixed slice, matched curves), apply it to hourly ERA5 over the held-out
2024, and compare against the CEN hourly series at three aggregations: hourly,
daily and monthly. The monthly row is a CONTROL: it must reproduce the known
0.1227 -> 0.1040, or the pipeline is miswired and the hourly numbers mean
nothing.

PRE-SPECIFIED GATES (written before any run):

  G1 (helps at all). Hourly corrected RMSE < hourly uncorrected RMSE.

  G2 (survives the timescale). The relative hourly RMSE reduction is at least
     HALF the monthly reduction. Chile monthly is 0.12271 -> 0.10398, a 15.3%
     reduction, so G2 requires >= 7.6% at hourly.

  G3 (bias). |hourly corrected MBE| < |hourly uncorrected MBE|.

PREDICTION (recorded before running, so it cannot be retrofitted): the affine
correction is a LEVEL correction, one scalar and offset per cluster, fitted by
matching monthly mean capacity factors. Most hourly error is within-month timing
and shape, which no per-cluster level correction can touch. So MBE should
improve substantially, RMSE much less than at monthly, and correlation barely at
all: G1 and G3 pass, G2 fails. If G2 PASSES, the correction is doing more than
level correction, and that is the more interesting result.

Time alignment is load-bearing here in a way it is not monthly: CEN publishes a
fixed UTC-4 with no DST and ERA5 is UTC, so a missed shift would be a 4-hour
misalignment that inflates hourly error and invalidates the test. The repo's own
converter is used.

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/analysis/hourly_resolution_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("PYVWF_INPUT", "input/combined")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import vwf.wind as wind  # noqa: E402
from vwf.clustering import cluster_turbines  # noqa: E402
from vwf.config import PyVWFPaths  # noqa: E402
from vwf.data import load_power_curves  # noqa: E402
from vwf.datasets.cen_cl import _local_to_utc, wind_rows  # noqa: E402
from vwf.datasets.era5 import prep_era5  # noqa: E402
from vwf.harness.corrections import get_correction  # noqa: E402
from vwf.harness.regions import load_region  # noqa: E402
from vwf.sources import get_source  # noqa: E402

warnings.simplefilter("ignore")

YEAR = 2024
NUM_CLU = 10
TIME_RES = "fixed"
TRAIN_RUN = Path("output/validation/cl_matched_2026-07-24/CL/train-matched")
RAW_CEN = Path("input/raw/cen")
MONTHLY_REFERENCE = (0.12271, 0.10398)  # known CL monthly uncorrected -> corrected


def _metrics(df: pd.DataFrame, sim: str) -> dict:
    d = df[sim] - df["obs"]
    r = float(np.corrcoef(df[sim], df["obs"])[0, 1]) if len(df) > 1 else float("nan")
    return {
        "n": len(df),
        "mbe": float(d.mean()),
        "mae": float(d.abs().mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "r": r,
    }


def _simulate_year(spec, clus_info, power_curves, factors, model, *, daily: bool):
    """Uncorrected and corrected CF for the year, one ERA5 month at a time.

    ``daily=True`` reproduces the published method: ERA5 is averaged to daily
    means BEFORE the power curve. ``daily=False`` keeps the file's native hourly
    resolution and applies the curve per hour. The two are different models, not
    two views of one, because the curve is convex.

    ERA5 is loaded per month so peak memory stays near one month of the Chile
    box; the simulated output is only (times x plants) and is small enough to
    hold for the whole year.
    """
    era5_dir = PyVWFPaths.INPUT_ROOT / spec.era5_path
    unc_parts, cor_parts = [], []
    for month in range(1, 13):
        matches = sorted(era5_dir.glob(f"*{YEAR}_{month:02d}.nc"))
        if not matches:
            print(f"  month {month:02d}: no ERA5 file, skipped", flush=True)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            os.symlink(matches[0].resolve(), Path(tmp) / matches[0].name)
            rea = prep_era5(spec.code, False, True, bbox=spec.bbox,
                            era5_dir=Path(tmp), resample_daily=daily)
        _, unc = wind.simulate_wind(rea, clus_info, power_curves)
        _, cor = model.apply(
            rea, clus_info, power_curves, factors, TIME_RES, seasons=spec.seasons
        )
        unc_parts.append(unc)
        cor_parts.append(cor)
        del rea
        print(f"  month {month:02d}: {len(unc)} steps", flush=True)
    return (pd.concat(unc_parts, ignore_index=True),
            pd.concat(cor_parts, ignore_index=True))


def _hourly_observations() -> pd.DataFrame:
    """CEN hourly wind capacity factors for YEAR, on naive UTC timestamps."""
    frames = []
    for month in range(1, 13):
        path = RAW_CEN / f"cen_gen_{YEAR}_{month:02d}.json"
        if not path.exists():
            continue
        frames.append(pd.DataFrame(json.load(open(path))))
    gen = pd.concat(frames, ignore_index=True)
    w = wind_rows(gen).copy()

    cap = pd.to_numeric(w["potencia_maxima"], errors="coerce")
    mw = pd.to_numeric(w["gen_real_mw"], errors="coerce")
    w = w[(cap > 0) & mw.notna()].copy()
    w["obs"] = (mw[w.index] / cap[w.index]).clip(lower=0.0)
    w["ID"] = w["id_central"].astype(str).str.strip()
    w["time"] = _local_to_utc(pd.to_datetime(w["fecha_hora"]))
    return w[["ID", "time", "obs"]]


def _long(cf: pd.DataFrame, name: str) -> pd.DataFrame:
    out = cf.melt(id_vars="time", var_name="ID", value_name=name)
    out["ID"] = out["ID"].astype(str)
    out["time"] = pd.to_datetime(out["time"])
    return out


def main() -> int:
    spec = load_region(Path("configs/regions/cl.toml"))
    source = get_source(spec.source, spec.code)

    turb_info = source.load_metadata()
    power_curves = load_power_curves()
    train_fleet = pd.read_csv(TRAIN_RUN / f"train_turb_info_{NUM_CLU}.csv")
    factors = pd.read_csv(TRAIN_RUN / f"factors_{TIME_RES}_{NUM_CLU}.csv")
    clus_info = cluster_turbines(NUM_CLU, train_fleet, False, turb_info)
    model = get_correction(spec.correction_model)

    print(f"CL fleet: {len(clus_info)} plants, {NUM_CLU} clusters, {TIME_RES} slice")
    print("simulating at NATIVE HOURLY resolution")
    unc_h, cor_h = _simulate_year(
        spec, clus_info, power_curves, factors, model, daily=False)
    print("simulating at PUBLISHED DAILY resolution (wind averaged before the curve)")
    unc_d, cor_d = _simulate_year(
        spec, clus_info, power_curves, factors, model, daily=True)

    obs = _hourly_observations()
    print(f"CEN hourly rows: {len(obs):,} across {obs['ID'].nunique()} plants")

    hourly = (
        _long(unc_h, "unc").merge(_long(cor_h, "cor"), on=["time", "ID"])
        .merge(obs, on=["time", "ID"], how="inner")
        .dropna(subset=["unc", "cor", "obs"])
    )
    print(f"paired HOURLY rows: {len(hourly):,} across {hourly['ID'].nunique()} plants")
    if len(hourly) < 100_000:
        print("WARNING: far fewer paired hourly rows than the ~500k expected; "
              "check the time convention before trusting anything below.")

    def _agg(df, freq):
        k = df["time"].dt.floor("D") if freq == "D" else df["time"].dt.to_period("M")
        return df.assign(k=k).groupby(["ID", "k"], as_index=False)[
            ["unc", "cor", "obs"]].mean()

    obs_daily = _agg(obs.assign(unc=np.nan, cor=np.nan), "D")[["ID", "k", "obs"]]
    daily_pub = (
        _long(unc_d, "unc").merge(_long(cor_d, "cor"), on=["time", "ID"])
        .assign(k=lambda d: d["time"].dt.floor("D"))
        .merge(obs_daily, on=["ID", "k"], how="inner")
        .dropna(subset=["unc", "cor", "obs"])
    )

    # MUST-DISTINGUISH diagnostic. Two competing explanations for any hourly
    # degradation: (a) a general effect, the level correction plus the convex
    # curve, which would hit every plant; or (b) a few DEGENERATE clusters, the
    # known Atacama artifact (this factors file has scalars of 80.2 and 28.7),
    # whose absurd wind multipliers saturate the curve at hourly resolution
    # while monthly averaging masked them. Excluding the degenerate clusters
    # separates the two: if the rest is fine, the problem is (b), not (a).
    degenerate = set(
        factors.loc[factors["scalar"] > 3.0, "cluster"].astype(int).tolist()
    )
    ok_ids = set(
        clus_info.loc[~clus_info["cluster"].isin(degenerate), "ID"].astype(str)
    )
    hourly_ok = hourly[hourly["ID"].isin(ok_ids)]
    print(f"degenerate clusters (scalar > 3.0): {sorted(degenerate)}; "
          f"hourly rows retained {len(hourly_ok):,} of {len(hourly):,}")

    rows = []
    for label, frame in [
        ("hourly (native)", hourly),
        ("hourly, degenerate clusters dropped", hourly_ok),
        ("daily (from hourly power)", _agg(hourly, "D")),
        ("daily (published: daily wind)", daily_pub),
        ("monthly (from hourly power)", _agg(hourly, "M")),
        ("monthly (published: daily wind)",
         daily_pub.assign(time=daily_pub["k"]).pipe(
             lambda d: d.assign(k=pd.to_datetime(d["k"]).dt.to_period("M"))
         ).groupby(["ID", "k"], as_index=False)[["unc", "cor", "obs"]].mean()),
    ]:
        u, c = _metrics(frame, "unc"), _metrics(frame, "cor")
        rows.append({
            "arm": label, "n": u["n"],
            "unc_rmse": u["rmse"], "cor_rmse": c["rmse"],
            "rmse_red_%": 100 * (1 - c["rmse"] / u["rmse"]),
            "unc_mbe": u["mbe"], "cor_mbe": c["mbe"],
            "unc_r": u["r"], "cor_r": c["r"],
        })
    table = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("RAW TABLE: uncorrected vs corrected, by resolution arm (CL 2024)")
    print("=" * 100)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    h = table[table["arm"] == "hourly (native)"].iloc[0]
    m_pub = table[table["arm"] == "monthly (published: daily wind)"].iloc[0]
    m_hr = table[table["arm"] == "monthly (from hourly power)"].iloc[0]
    d_pub = table[table["arm"] == "daily (published: daily wind)"].iloc[0]
    d_hr = table[table["arm"] == "daily (from hourly power)"].iloc[0]
    monthly_red = 100 * (1 - MONTHLY_REFERENCE[1] / MONTHLY_REFERENCE[0])
    g1 = h["cor_rmse"] < h["unc_rmse"]
    g2 = h["rmse_red_%"] >= monthly_red / 2
    g3 = abs(h["cor_mbe"]) < abs(h["unc_mbe"])

    print(f"\nCONTROL: published-method monthly {m_pub['unc_rmse']:.4f} -> "
          f"{m_pub['cor_rmse']:.4f}  (canonical {MONTHLY_REFERENCE[0]:.4f} -> "
          f"{MONTHLY_REFERENCE[1]:.4f})")
    print("  Exact agreement is NOT expected: the canonical run builds its monthly")
    print("  observations through the processed cl_obs.csv path (capacity overrides,")
    print("  commissioning-prefix stripping, exclusions), while this reads raw CEN.")
    print("  The control is for order of magnitude and sign, not equality.")
    print("\nCONVEXITY COST (same wind, curve applied before vs after averaging):")
    print(f"  daily   uncorrected RMSE  published {d_pub['unc_rmse']:.4f}  vs  "
          f"from-hourly {d_hr['unc_rmse']:.4f}")
    print(f"  monthly uncorrected MBE   published {m_pub['unc_mbe']:+.4f}  vs  "
          f"from-hourly {m_hr['unc_mbe']:+.4f}")
    print("\nPRE-SPECIFIED GATES")
    print(f"  G1 helps at all:       {h['cor_rmse']:.4f} < {h['unc_rmse']:.4f}"
          f"   [{'PASS' if g1 else 'FAIL'}]")
    print(f"  G2 survives timescale: {h['rmse_red_%']:.1f}% vs {monthly_red/2:.1f}% "
          f"required   [{'PASS' if g2 else 'FAIL'}]")
    print(f"  G3 bias reduced:       |{h['cor_mbe']:+.4f}| < |{h['unc_mbe']:+.4f}|"
          f"   [{'PASS' if g3 else 'FAIL'}]")

    out = Path("output/hourly_test")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "cl_2024_by_aggregation.csv", index=False)
    print(f"\nwrote {out/'cl_2024_by_aggregation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

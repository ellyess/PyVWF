#!/usr/bin/env python3
"""E1: leave-one-region-out transfer, physics-informed vs the incumbent.

The pre-specified experiment (docs/findings/method-physics-informed-prespecification.md). For each
region in turn: fit on the other four, apply zero-shot to the held-out region's
test year, and score capacity-factor error with the harness's own
``skill_metrics`` against observed generation.

Arms, all scored identically:

  uncorrected     ERA5 through the incumbent pipeline, no correction. The
                  baseline every gate is measured against.
  pinn            physics-informed correction fitted on the other four regions.
  pinn-ablation   identical features and parameter count, every physical
                  constraint removed (gate P3).
  rf-transfer     the incumbent statistical baseline: RandomForest on the
                  per-cluster affine factors, published SET_A features, trained
                  on the other four regions and applied per test unit (gate P2).
  pinn-in-region  fitted on the held-out region's OWN training years. Not a
                  transfer arm and not a gate -- a reference point for how much
                  of the achievable correction transfer recovers.
  affine-in-region  the published per-cluster affine numbers, quoted from the
                  validation scorecard for the same reason.

Run: PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python \
         scripts/pinn/e1_loro.py --seeds 0 1 2 3 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from vwf.harness.regions import load_region  # noqa: E402
from vwf.harness.skill import collapse_pseudo_replicates, skill_metrics  # noqa: E402
from vwf.pinn.physics import PowerCurveBank, expected_cf, hub_wind_ratio, monthly_mean  # noqa: E402
from vwf.pinn.train import (  # noqa: E402
    UNIT_BATCH, coverage_weight, fit, load_regions, predict_frame,
)
from analysis.ml_transfer_retest import (  # noqa: E402
    RF_KW, RUNS, SET_A, build_centroids, terrain_features,
)

REGIONS = ["DK", "DE", "UK", "US", "BR"]
CACHE = ROOT / "output" / "pinn" / "cache"
OUT = ROOT / "output" / "pinn" / "e1"
CONFIGS = ROOT / "configs" / "regions"

# Published per-cluster affine results, quoted for reference only, from
# docs/findings/scorecard.md (best held-out config).
AFFINE_SCORECARD = {"DK": 0.085, "DE": 0.057, "UK": 0.115, "US": 0.098, "BR": 0.105}


# ------------------------------------------------------------- RF baseline ---
# The published SET_A features depend only on position, so they are identical
# for every holdout, and deriving them means differentiating a 22-million-pixel
# ETOPO window for the American box. scripts/pinn/prep_rf_features.py does that
# once, in a process holding nothing else; this reads the result. Falling back
# to computing them inline keeps the script self-sufficient, at the cost of
# that memory spike.
RF_FEATURES = ROOT / "output" / "pinn" / "rf_features"
_CENTROIDS: pd.DataFrame | None = None
_UNIT_FEATURES: dict[str, pd.DataFrame] = {}


def _centroid_features() -> pd.DataFrame:
    global _CENTROIDS
    if _CENTROIDS is None:
        path = RF_FEATURES / "centroids.csv"
        if path.exists():
            _CENTROIDS = pd.read_csv(path)
        else:
            c = build_centroids(RUNS)
            for x in ("lon", "lat"):
                c[f"{x}_norm"] = (c[x] - c[x].min()) / (c[x].max() - c[x].min())
            _CENTROIDS = terrain_features(c)
    return _CENTROIDS


def _unit_features(holdout: str, test_meta: pd.DataFrame) -> pd.DataFrame:
    """SET_A features for one region's test fleet, aligned row-for-row.

    The cached file holds the FULL test fleet; the tensor fleet has had units
    without ERA5 coverage removed, so the two differ in length and the rows must
    be matched rather than assumed. Matching is on ID: an earlier version keyed
    on rounded coordinates, and because several units share a location exactly
    -- six in the Danish fleet alone -- the lookup returned 5,411 rows for a
    5,399-unit fleet and silently shifted every prediction after the first
    duplicate. The assertion below is what makes that impossible to repeat.
    """
    if holdout not in _UNIT_FEATURES:
        path = RF_FEATURES / f"units_{holdout}.csv"
        if path.exists():
            feats = pd.read_csv(path, dtype={"ID": str})
            if "ID" not in feats.columns:
                raise ValueError(
                    f"{path} predates ID-based alignment; rerun "
                    "scripts/pinn/prep_rf_features.py")
            feats = (test_meta[["ID"]].astype({"ID": str})
                     .merge(feats, on="ID", how="left", validate="one_to_one"))
            if len(feats) != len(test_meta) or feats[SET_A].isna().any().any():
                raise ValueError(
                    f"{holdout}: cached RF features do not align with the test "
                    f"fleet ({len(feats)} vs {len(test_meta)} rows, "
                    f"{int(feats[SET_A].isna().any(axis=1).sum())} unmatched)")
            _UNIT_FEATURES[holdout] = feats
        else:
            centroids = _centroid_features()
            units = test_meta[["lon", "lat"]].copy()
            units["region"] = holdout
            for x in ("lon", "lat"):
                lo, hi = centroids[x].min(), centroids[x].max()
                units[f"{x}_norm"] = (units[x] - lo) / (hi - lo)
            units["abs_lat"] = units["lat"].abs()
            _UNIT_FEATURES[holdout] = terrain_features(units)
    return _UNIT_FEATURES[holdout]


def rf_affine_predictions(holdout: str, test_meta: pd.DataFrame, seeds):
    """Per-unit (scalar, offset) from the published RF transfer recipe."""
    centroids = _centroid_features()
    train = centroids[centroids.region != holdout]
    units = _unit_features(holdout, test_meta)

    out = {}
    for target in ("scalar", "offset"):
        preds = [RandomForestRegressor(random_state=s, **RF_KW)
                 .fit(train[SET_A], train[target]).predict(units[SET_A])
                 for s in seeds]
        out[target] = np.mean(preds, axis=0)
    return out["scalar"], out["offset"]


@torch.no_grad()
def affine_frame(r, scalar: np.ndarray, offset: np.ndarray) -> pd.DataFrame:
    """Monthly CF with an affine wind-speed correction, incumbent-style."""
    a = torch.as_tensor(scalar, dtype=torch.float32)
    b = torch.as_tensor(offset, dtype=torch.float32)
    preds = []
    for start in range(0, r.n_units, UNIT_BATCH):
        sl = torch.arange(start, min(start + UNIT_BATCH, r.n_units))
        ratio = hub_wind_ratio(r.height[sl], z0=r.z0[:, sl], profile="log")
        u = (r.w[:, sl] * ratio) * a[sl] + b[sl]
        cf = expected_cf(u.clamp(min=0.0), None, r.curve_idx[sl], r.bank, None)
        preds.append(monthly_mean(cf, r.month_id, len(r.months)))
    pred = torch.cat(preds, dim=1).numpy()
    years = np.array([y for y, _ in r.months])
    months = np.array([m for _, m in r.months])
    M, N = pred.shape
    frame = pd.DataFrame({
        "ID": np.repeat(r.ids[None, :], M, axis=0).ravel(),
        "year": np.repeat(years[:, None], N, axis=1).ravel(),
        "month": np.repeat(months[:, None], N, axis=1).ravel(),
        "cf_sim": pred.ravel(),
        "cf_obs": r.obs.numpy().ravel(),
        "capacity": np.repeat(r.capacity.numpy()[None, :], M, axis=0).ravel(),
    })
    return frame.dropna(subset=["cf_obs"]).reset_index(drop=True)


def score(frame: pd.DataFrame, spec) -> dict:
    f = frame.dropna(subset=["cf_sim"])
    f = collapse_pseudo_replicates(f, spec)
    return skill_metrics(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear heads")
    ap.add_argument("--regions", nargs="+", default=REGIONS,
                    help="regions to hold out and score, one at a time")
    ap.add_argument("--train-pool", nargs="+", default=REGIONS,
                    help="pool the training regions are drawn from, minus the "
                         "holdout; widen it for the nine-region test (E7)")
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--arms", nargs="+",
                    default=["pinn", "pinn-ablation", "pinn-in-region"],
                    choices=["pinn", "pinn-ablation", "pinn-in-region",
                             "pinn-abstain"],
                    help="which fitted arms to run; the default is the "
                         "pre-specified E1 set")
    ap.add_argument("--profile", default="power",
                    choices=["power", "shear-log", "log"],
                    help="hub-height profile form (addendum 9)")
    ap.add_argument("--wake", action="store_true",
                    help="add the hyperbolic array-loss term (addendum 8)")
    ap.add_argument("--density", action="store_true",
                    help="apply the ISA air-density correction (addendum 1); "
                         "OFF for the pre-specified E1 arms")
    args = ap.parse_args()
    hidden = args.hidden or None

    OUT.mkdir(parents=True, exist_ok=True)
    pool = list(dict.fromkeys([*args.train_pool, *args.regions]))
    specs = {c: load_region(CONFIGS / f"{c.lower().replace('-', '_')}.toml")
             for c in pool}
    train_sets = {c: load_regions([c], "train", CACHE, quiet=True)[0] for c in pool}
    test_sets = {c: load_regions([c], "test", CACHE, quiet=True)[0]
                 for c in args.regions}
    print("caches loaded:",
          {c: f"{train_sets[c].n_units}tr" for c in pool}, flush=True)

    rows, physics_records = [], []
    for holdout in args.regions:
        t0 = time.time()
        others = [c for c in args.train_pool if c != holdout]
        te, spec = test_sets[holdout], specs[holdout]
        print(f"\n=== holdout {holdout}  (train on {'+'.join(others)}) ===")

        base = score(predict_frame(te, None, None), spec)
        rows.append(dict(holdout=holdout, arm="uncorrected", seed=-1, **base))
        print(f"  uncorrected      RMSE {base['rmse']:.4f}  MBE {base['mbe']:+.4f}",
              flush=True)

        if set(args.train_pool) <= set(REGIONS) and holdout in REGIONS:
            # The published RF recipe is defined on the five canonical training
            # runs; with a widened pool there is no matching baseline, so it is
            # skipped rather than quietly compared against something else.
            sc, off = rf_affine_predictions(holdout, te_meta_frame(te), args.seeds)
            rf = score(affine_frame(te, sc, off), spec)
            rows.append(dict(holdout=holdout, arm="rf-transfer", seed=-1, **rf))
            print(f"  rf-transfer      RMSE {rf['rmse']:.4f}  MBE {rf['mbe']:+.4f}  "
                  f"(scalar {sc.mean():.3f}+-{sc.std():.3f})", flush=True)

        arm_specs = {
            # name             physics  training regions  damp outside envelope
            "pinn":           (True,  others,    False),
            "pinn-ablation":  (False, others,    False),
            "pinn-in-region": (True,  [holdout], False),
            "pinn-abstain":   (True,  others,    True),
        }
        for arm in args.arms:
            physics, train_codes, abstain = arm_specs[arm]
            for seed in args.seeds:
                tr = [train_sets[c] for c in train_codes]
                # Density is part of the model's physics, so the ablation arm
                # never gets it: P3 must stay a clean physics/no-physics test.
                dens = args.density and physics
                model, std, hist = fit(tr, hidden=hidden, physics=physics,
                                       profile=args.profile, density=dens,
                                       wake=args.wake and physics,
                                       epochs=args.epochs, seed=seed, verbose=False)
                damp = coverage_weight(te, tr, std) if abstain else None
                m = score(predict_frame(te, model, std, density=dens,
                                        damp=damp, profile=args.profile), spec)
                rows.append(dict(holdout=holdout, arm=arm, seed=seed, **m))
                if arm == "pinn":
                    rep = model.report(std.terrain(te), std.fleet(te), te.relief,
                                       te.capdens)
                    # Pre-registered prediction 1 (addendum 1): the speed-up at
                    # high-elevation sites should FALL once density is modelled
                    # explicitly, because it is currently absorbing the deficit.
                    with torch.no_grad():
                        g, *_ = model(std.terrain(te), std.fleet(te),
                                      te.relief, te.capdens)
                        high = te.elevation > 800.0
                        rep["speedup_high_elev"] = (
                            float(torch.exp(g[high]).mean()) if bool(high.any())
                            else float("nan"))
                        rep["n_high_elev"] = int(high.sum())
                    physics_records.append(dict(holdout=holdout, seed=seed,
                                                density=dens,
                                                final_loss=hist[-1], **rep))
            sub = [r for r in rows if r["holdout"] == holdout and r["arm"] == arm]
            rm = np.array([r["rmse"] for r in sub])
            mb = np.array([r["mbe"] for r in sub])
            print(f"  {arm:16s} RMSE {rm.mean():.4f} +- {rm.std():.4f}  "
                  f"MBE {mb.mean():+.4f}", flush=True)
        # Written after every holdout so a long run can be inspected, and
        # resumed from, while it is still going.
        pd.DataFrame(rows).to_csv(OUT / f"e1_{args.tag}_raw.csv", index=False)
        pd.DataFrame(physics_records).to_csv(
            OUT / f"e1_{args.tag}_physics.csv", index=False)
        print(f"  [{time.time()-t0:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"e1_{args.tag}_raw.csv", index=False)
    pd.DataFrame(physics_records).to_csv(OUT / f"e1_{args.tag}_physics.csv", index=False)

    summarise(res, args.tag)


def te_meta_frame(r) -> pd.DataFrame:
    """ID/lon/lat frame for the RF baseline, in the tensor unit order."""
    return pd.DataFrame({"ID": r.ids.astype(str), "lon": r.lon, "lat": r.lat})


def summarise(res: pd.DataFrame, tag: str):
    pd.set_option("display.width", 220)
    agg = (res.groupby(["holdout", "arm"])
              .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                   mbe=("mbe", "mean"), r=("pearson_r", "mean"))
              .reset_index())
    piv = agg.pivot(index="holdout", columns="arm", values="rmse")
    print(f"\n{'='*100}\n### RMSE by holdout region and arm (capacity-weighted, test year)\n")
    print(piv.round(4).to_string())

    base = piv["uncorrected"]
    print(f"\n### Skill against uncorrected ERA5 (positive = better)\n")
    skill = 1 - piv.div(base, axis=0) ** 2
    print(skill.round(3).to_string())
    agg.to_csv(OUT / f"e1_{tag}_summary.csv", index=False)

    # The gates are defined on the "pinn" arm. A run of a different arm -- the
    # abstention variant, say -- has no such column, and its per-holdout results
    # are already on disk by this point, so the summary must decline to score
    # rather than take the process down after an hour of fitting.
    if "pinn" not in piv.columns:
        print(f"\n### Gates not scored: this run has no 'pinn' arm "
              f"(arms present: {', '.join(piv.columns)})")
        print(f"    per-holdout results are in e1_{tag}_raw.csv; score them with "
              f"scripts/pinn/e1_report.py --tag {tag}")
        return

    print(f"\n### Gate P1: pinn beats uncorrected, and degrades nothing by >10%\n")
    better = (piv["pinn"] < base)
    worse10 = (piv["pinn"] > base * 1.10)
    print(f"  better in {int(better.sum())}/5: {', '.join(sorted(piv.index[better]))}")
    print(f"  degraded >10% in {int(worse10.sum())}: "
          f"{', '.join(sorted(piv.index[worse10])) or 'none'}")
    print(f"  P1 {'PASS' if better.sum() >= 3 and not worse10.any() else 'FAIL'}")

    if "rf-transfer" in piv and "pinn" in piv:
        beats = (piv["pinn"] < piv["rf-transfer"])
        print(f"\n### Gate P2: pinn beats the incumbent RF transfer\n")
        print(f"  beats rf-transfer in {int(beats.sum())}/5: "
              f"{', '.join(sorted(piv.index[beats]))}")
        print(f"  P2 {'PASS' if beats.sum() >= 3 else 'FAIL'}")

    if "pinn-ablation" in piv and "pinn" in piv:
        print(f"\n### Gate P3: does the physics earn its place?\n")
        d = (piv["pinn-ablation"] - piv["pinn"])
        print(d.round(4).to_string())
        print(f"  physics better in {int((d>0).sum())}/5 regions "
              f"(positive = ablation worse)")


if __name__ == "__main__":
    main()

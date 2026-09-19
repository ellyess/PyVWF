#!/usr/bin/env python3
"""Compare rebuilt physics-informed caches with the ones the published run used.

Read-only on both. For each region and split it reports whether the two caches
hold the same units, the same observations and the same fleet metadata, and the
largest absolute difference in each daily field over the units and days both
hold. A position finite in one cache and missing in the other is counted, not
folded into the difference.

This is the input check a rerun is gated on before any model is fitted: if the
winds agree, a difference in a fitted arm is a difference of code or of the
roughness treatment, not of data.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/g0_cache_reproduction.py \
         --old <published cache root> --new <rebuilt cache root> --out <dir>
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vwf.pinn.cache import load_cache  # noqa: E402

FIELDS = ("w_mean", "w_std", "shear", "z0")
META = ("lon", "lat", "capacity", "height", "model")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(old, new) -> dict:
    """One region/split's comparison, as a flat record."""
    old_ids = old.meta["ID"].astype(str).tolist()
    new_ids = new.meta["ID"].astype(str).tolist()
    common = sorted(set(old_ids) & set(new_ids))
    o_pos = {i: k for k, i in enumerate(old_ids)}
    n_pos = {i: k for k, i in enumerate(new_ids)}
    oi = np.array([o_pos[i] for i in common], dtype=int)
    ni = np.array([n_pos[i] for i in common], dtype=int)

    rec = {
        "units_old": len(old_ids),
        "units_new": len(new_ids),
        "units_common": len(common),
        "units_only_old": len(set(old_ids) - set(new_ids)),
        "units_only_new": len(set(new_ids) - set(old_ids)),
        "days_old": len(old.dates),
        "days_new": len(new.dates),
    }
    days = old.dates.intersection(new.dates)
    rec["days_common"] = len(days)
    od = old.dates.get_indexer(days)
    nd = new.dates.get_indexer(days)

    om = old.meta.set_index(old.meta["ID"].astype(str)).loc[common]
    nm = new.meta.set_index(new.meta["ID"].astype(str)).loc[common]
    for col in META:
        if col not in om or col not in nm:
            rec[f"meta_{col}_mismatches"] = None
            continue
        if col == "model":
            rec[f"meta_{col}_mismatches"] = int((om[col].astype(str) != nm[col].astype(str)).sum())
        else:
            a, b = om[col].to_numpy(dtype=float), nm[col].to_numpy(dtype=float)
            rec[f"meta_{col}_mismatches"] = int(
                (~np.isclose(a, b, rtol=0, atol=1e-9) & ~(np.isnan(a) & np.isnan(b))).sum()
            )

    keys = ["ID", "year", "month"]
    oo = old.obs.astype({"ID": str}).dropna(subset=["obs"])
    no = new.obs.astype({"ID": str}).dropna(subset=["obs"])
    merged = oo.merge(no, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    rec["obs_rows_common"] = int(len(both))
    rec["obs_rows_only_old"] = int((merged["_merge"] == "left_only").sum())
    rec["obs_rows_only_new"] = int((merged["_merge"] == "right_only").sum())
    rec["obs_max_abs_diff"] = (
        float((both["obs_old"] - both["obs_new"]).abs().max()) if len(both) else None
    )

    for name in FIELDS:
        a = np.asarray(getattr(old, name))[np.ix_(od, oi)].astype("float64")
        b = np.asarray(getattr(new, name))[np.ix_(nd, ni)].astype("float64")
        fa, fb = np.isfinite(a), np.isfinite(b)
        rec[f"{name}_finite_mismatch"] = int((fa != fb).sum())
        both_finite = fa & fb
        rec[f"{name}_max_abs_diff"] = (
            float(np.abs(a - b)[both_finite].max()) if both_finite.any() else None
        )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="cache root the published run used")
    ap.add_argument("--new", required=True, help="rebuilt cache root")
    ap.add_argument("--out", required=True, help="directory for the comparison table")
    ap.add_argument("--regions", nargs="+", default=["DK", "DE", "UK", "US", "BR"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    rows = []
    for code in args.regions:
        for split in args.splits:
            old_dir = Path(args.old) / f"{code}_{split}"
            old = load_cache(code, split, args.old)
            new = load_cache(code, split, args.new)
            rec = {
                "region": code,
                "split": split,
                "old_fields_sha256": _sha256(old_dir / "fields.npz"),
                **compare(old, new),
            }
            rows.append(rec)
            print(
                f"[{code}/{split}] units {rec['units_old']} -> {rec['units_new']} "
                f"({rec['units_common']} common); obs max diff {rec['obs_max_abs_diff']}; "
                + "; ".join(f"{f} {rec[f'{f}_max_abs_diff']}" for f in FIELDS),
                flush=True,
            )
            del old, new

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "g0_cache_reproduction.csv", index=False)
    print(f"-> {out / 'g0_cache_reproduction.csv'}")


if __name__ == "__main__":
    main()

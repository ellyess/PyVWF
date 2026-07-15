#!/usr/bin/env python
"""D1 regression diff: compare two run directories at frame level.

Compares ``factors_*.csv`` and ``cor_cf_*.csv`` between a REFERENCE directory
(generated from the legacy ``main`` path) and a HARNESS directory, cell by
cell over numeric columns, and reports a pass/fail table against a tolerance.

The gate sits on factors and corrected-CF frames, not on metric numbers:
legacy ``vwf.metrics`` and harness ``vwf.harness.skill`` compute skill with
different (intentional) formulas, so a metric diff is a formula choice, not a
regression. If the frames match, the refactor preserved the method.

    python scripts/d1_regression.py --reference REF_DIR --harness RUN_DIR \
        --atol 1e-12 --label DK

Exit code 0 = every frame within tolerance, 1 = at least one FAIL, 2 = no
frames found to compare.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame):
    """(max_abs_diff, note). max is None on a structural mismatch, inf if the
    NaN pattern differs (which is itself a regression)."""
    if list(a.columns) != list(b.columns):
        return None, f"columns differ ({list(a.columns)} vs {list(b.columns)})"
    if a.shape != b.shape:
        return None, f"shapes differ ({a.shape} vs {b.shape})"
    num = a.select_dtypes(include=[np.number]).columns
    if len(num) == 0:
        return 0.0, "no numeric columns"
    an, bn = a[num].to_numpy(float), b[num].to_numpy(float)
    if (np.isnan(an) != np.isnan(bn)).any():
        return float("inf"), "NaN pattern differs"
    diff = np.where(np.isnan(an) & np.isnan(bn), 0.0, np.abs(an - bn))
    return (float(np.nanmax(diff)) if diff.size else 0.0), ""


def compare_dirs(reference: Path, harness: Path, atol: float, label: str):
    """Return (rows, any_fail). rows are dicts for the report table."""
    names = sorted(
        {p.name for pat in ("factors_*.csv", "cor_cf_*.csv") for p in reference.glob(pat)}
    )
    rows, any_fail = [], False
    for name in names:
        ref_f, run_f = reference / name, harness / name
        if not run_f.is_file():
            rows.append({"frame": name, "max_abs_diff": None, "status": "MISSING",
                         "note": "absent in harness output"})
            any_fail = True
            continue
        mad, note = max_abs_diff(pd.read_csv(ref_f), pd.read_csv(run_f))
        if mad is None:
            status = "STRUCT"
            any_fail = True
        elif mad <= atol:
            status = "ok"
        else:
            status = "FAIL"
            any_fail = True
        rows.append({"frame": name, "max_abs_diff": mad, "status": status, "note": note})
    return rows, any_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--atol", type=float, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None, help="Optional CSV report path")
    args = ap.parse_args()

    reference, harness = Path(args.reference), Path(args.harness)
    rows, any_fail = compare_dirs(reference, harness, args.atol, args.label)
    if not rows:
        print(f"[{args.label}] NO frames found in {reference}")
        sys.exit(2)

    worst = max((r["max_abs_diff"] for r in rows
                 if isinstance(r["max_abs_diff"], float) and np.isfinite(r["max_abs_diff"])),
                default=0.0)
    for r in rows:
        mad = r["max_abs_diff"]
        mad_s = "   n/a   " if mad is None else f"{mad:.3e}"
        extra = f"  ({r['note']})" if r["note"] else ""
        print(f"[{args.label}] {r['frame']:<24} diff={mad_s}  {r['status']}{extra}")
    verdict = "FAIL" if any_fail else "PASS"
    print(f"[{args.label}] worst={worst:.3e}  atol={args.atol:.0e}  ->  {verdict}")

    if args.out:
        df = pd.DataFrame(rows)
        df.insert(0, "label", args.label)
        df.to_csv(args.out, index=False)

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figures for the physics-informed correction: transfer, coverage, physics.

Three panels, each answering one question the tables answer less legibly:

  1. What the correction does to held-out capacity-factor error, per region and
     per arm, against the incumbent affine fit as a reference line.
  2. Whether transfer difficulty was predictable in advance, by plotting skill
     against the physiographic coverage D5 measured before any model ran.
  3. What the model learned: fitted speed-up against ERA5-cell relief, which is
     the relationship the whole parameterisation rests on.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/e1_figures.py --tag primary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "pinn"))

from vwf.viz.palettes import OKABE_ITO  # noqa: E402
from vwf.viz.style import plot_style, savefig  # noqa: E402

E1 = ROOT / "output" / "pinn" / "e1"
D5 = ROOT / "output" / "pinn" / "d5"
E2 = ROOT / "output" / "pinn" / "e2"
FIG = ROOT / "output" / "pinn" / "figures"
REGIONS = ["DK", "DE", "UK", "US", "BR"]

ARM_STYLE = {
    "uncorrected": ("uncorrected ERA5", "#7f7f7f"),
    "rf-transfer": ("RF transfer (incumbent ML)", OKABE_ITO[1]),
    "pinn": ("physics-informed, zero-shot", OKABE_ITO[4]),
    "pinn-abstain": ("physics-informed + abstention", OKABE_ITO[2]),
    "pinn-ablation": ("ablation: constraints removed", OKABE_ITO[5]),
    "pinn-in-region": ("physics-informed, in-region", OKABE_ITO[0]),
}


def fig_transfer(raw: pd.DataFrame, table: pd.DataFrame, out: Path):
    """Held-out RMSE by region and arm, with the incumbent affine as a marker."""
    arms = [a for a in ARM_STYLE if a in raw.arm.unique()]
    regions = [r for r in REGIONS if r in raw.holdout.unique()]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    width = 0.8 / len(arms)
    x = np.arange(len(regions))
    for i, arm in enumerate(arms):
        sub = raw[raw.arm == arm].groupby("holdout").rmse
        m = np.array([sub.mean().get(r, np.nan) for r in regions])
        e = np.array([sub.std().get(r, 0.0) or 0.0 for r in regions])
        label, colour = ARM_STYLE[arm]
        ax.bar(x + i * width - 0.4 + width / 2, m, width, yerr=e,
               label=label, color=colour, edgecolor="none",
               error_kw=dict(lw=0.6, capsize=1.5))
    if "affine_best" in table:
        aff = [table.loc[r, "affine_best"] if r in table.index else np.nan
               for r in regions]
        ax.plot(x, aff, "k_", markersize=18, markeredgewidth=1.2,
                label="incumbent affine, in-region (best of sweep)")
    ax.set_xticks(x, regions)
    ax.set_ylabel("held-out capacity-factor RMSE")
    ax.set_xlabel("region held out of training")
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out / "e1_transfer_by_region.pdf")


def fig_coverage(table: pd.DataFrame, out: Path):
    """Skill against the physiographic coverage measured before any model ran."""
    # D5 writes a tagged file per region set; fall back to the untagged name
    # written before tagging existed, and skip the panel rather than crash the
    # whole figure run if neither is there.
    candidates = [D5 / "d5_joint_five.csv", D5 / "d5_joint.csv", D5 / "d5_joint_nine.csv"]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        print("  (skipping coverage panel: no D5 joint-coverage file found)")
        return
    joint = pd.read_csv(path).set_index("holdout")
    regions = [r for r in REGIONS if r in table.index and r in joint.index]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for arm, marker in (("pinn", "o"), ("rf-transfer", "s")):
        if arm not in table:
            continue
        skill = 1 - (table.loc[regions, arm] / table.loc[regions, "uncorrected"]) ** 2
        label, colour = ARM_STYLE[arm]
        ax.scatter(joint.loc[regions, "joint_covered"], skill, marker=marker,
                   s=22, color=colour, label=label, zorder=3)
        for r in regions:
            s = 1 - (table.loc[r, arm] / table.loc[r, "uncorrected"]) ** 2
            ax.annotate(r, (joint.loc[r, "joint_covered"], s), fontsize=5,
                        xytext=(3, 2), textcoords="offset points")
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("share of held-out units inside the training\nphysiographic envelope")
    ax.set_ylabel("skill against uncorrected ERA5")
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out / "e1_skill_vs_coverage.pdf")


def fig_physics(out: Path):
    """Fitted speed-up against ERA5-cell relief, the relationship it rests on."""
    path = E2 / "e2_fitted_physics.csv"
    if not path.exists():
        print(f"  (skipping physics panel: {path} not written yet)")
        return
    df = pd.read_csv(path)
    df = df[~df.density] if "density" in df else df
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for i, (code, sub) in enumerate(df.groupby("region")):
        ax.scatter(sub.relief_28km, sub.speedup, s=3, alpha=0.35,
                   color=OKABE_ITO[i % len(OKABE_ITO)], label=code,
                   edgecolors="none", rasterized=True)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("elevation range within the ERA5 cell (m)")
    ax.set_ylabel("fitted wind-speed speed-up")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=5.5, markerscale=2.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out / "e1_speedup_vs_relief.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="primary")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    plot_style()

    raw = pd.read_csv(E1 / f"e1_{args.tag}_raw.csv")
    tpath = E1 / f"e1_{args.tag}_table.csv"
    table = (pd.read_csv(tpath, index_col=0) if tpath.exists()
             else raw.groupby(["holdout", "arm"]).rmse.mean().unstack())
    fig_transfer(raw, table, FIG)
    fig_coverage(table, FIG)
    fig_physics(FIG)
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()

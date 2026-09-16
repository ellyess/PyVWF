# Re-running the physics-informed headline on today's pipeline: registered plan and gates

**Date:** 2026-09-16 (registered on commit, before any cache is rebuilt and
before any model is fitted)
**Scope:** whether the leave-one-region-out result in
`method-physics-informed.md` reproduces from a committed code state, and
whether its gates still pass once the pipeline changes since 2026-08-23 are
applied. Terms follow `CONTEXT.md`. The original gates are in
`method-physics-informed-prespecification.md`.

**No published physics-informed figure is attributable to a commit.** The
primary run's log records commit `bd1c721`, which no branch contains, and
every later run recorded no commit at all. The results sit in a session
worktree, `.claude/worktrees/era5-wind-bias-correction-88a2df/output/pinn/`.
This plan re-runs the headline once, from a clean tree, with a manifest, and
fixes in advance what each outcome does to the published document.

**Everything below is fixed before any rerun exists.** A condition or a gate
added later is labelled post hoc and cannot pass.

## What is re-run, and what is not

**Re-run:** the primary leave-one-region-out comparison, E1, over DK, DE, UK,
US and BR, with the published arms and settings:

| Setting | Value |
|---|---|
| Arms | `uncorrected`, `pinn`, `pinn-ablation`, `pinn-in-region` |
| Seeds | 0, 1, 2, 3, 42 |
| Epochs | 60 |
| Heads | linear (`--hidden 0`) |
| Profile | power law on the measured shear (`--profile power`) |
| Density, wake, bound scale | off, off, 1.0 |
| Curve library | `input/combined`, the licensed library |

**Not re-run, and their figures stay unattributable:** the fresh-region gate
on AR, AU-NEM, CL and NZ (Q1, Q2), the diagnostics D0 to D8, the physics audit
E2, and the sensitivities E3 to E11. This plan covers the headline only.

## What changes from the published run

Four changes, each declared here because each could move a number.

1. **Configurations.** Each region loads its scorecard config, not its
   maintained config: `dk_k100.toml`, `de_k100.toml`, `uk_k50.toml`,
   `us_k250.toml` and `br_k60.toml` under `configs/regions/scorecard/`. For
   DK, DE and UK this changes the ERA5 path from `era5/EU` to
   `era5/EU_2026-09` and the roughness to `derived`. The physics-informed path
   reads neither the cluster count nor the time slice, so the other
   differences between the two sets of configs do nothing here.
2. **Roughness.** DK, DE and UK move from the annual-mean roughness to the
   per-timestep roughness. US and BR already derived it per timestep from
   their hourly files, and still do. **The `pinn`, `pinn-ablation` and
   `pinn-in-region` arms never read the roughness**: the power-law profile
   uses the measured 10 to 100 m shear. Only the `uncorrected` arm, which
   applies the neutral log law on the roughness, can move for this reason.
3. **Scoring.** Every arm of a holdout is scored on the rows every arm can
   score, as the harness scores its variants, and the excluded rows are
   written out. The power-curve bank clamps a speed outside its table to the
   table's end value rather than returning a missing value, so the share of
   capacity-weighted unit-days with an off-curve value is tallied per
   condition and reported.
4. **The `rf-transfer` arm is not run, and gate P2 is not re-gated.** Its
   training targets are the July factor files named in
   `scripts/analysis/ml_transfer_retest.py`, fitted on the annual-mean
   roughness for DK, DE and UK. Applied to per-timestep winds, it would score
   factors fitted under one treatment on winds simulated under another, which
   is not the comparison P2 made. A fair P2 needs the five factor sets refitted
   on the per-timestep treatment, which belongs with port phase 3, the
   machine-learning module.

The code that makes these changes is committed before this document:
`vwf.pinn.era5_stats` and `vwf.pinn.cache` apply and record the treatment the
config requests and the loaded extent; `vwf.pinn.train` records units dropped
for having no wind and tallies off-curve values; `scripts/pinn/e1_loro.py`
scores on common rows and writes a manifest; and
`scripts/pinn/g0_cache_reproduction.py` compares caches.

**The physics-informed path never extrapolates winds.** A unit outside the
loaded extent gets no wind and is dropped from every arm alike, whatever the
config's `allow_extrapolation` says. DK's scorecard config opts in, so the
scorecard's DK row simulates Bornholm and this run does not. Within this run
every arm sees the same fleet, so no comparison here is affected. The dropped
units and their capacity share are recorded per region and split.

## Commands

Run from the repository root, on a clean tree, in this order, one process at a
time. Each script refuses a dirty tree.

```bash
PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python -u \
  scripts/pinn/build_cache.py --regions DK DE UK US BR \
  --out output/pinn_rerun_2026-09-16/cache \
  --config DK=configs/regions/scorecard/dk_k100.toml \
  --config DE=configs/regions/scorecard/de_k100.toml \
  --config UK=configs/regions/scorecard/uk_k50.toml \
  --config US=configs/regions/scorecard/us_k250.toml \
  --config BR=configs/regions/scorecard/br_k60.toml \
  --registration docs/findings/method-physics-informed-rerun-prereg.md
```

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -u scripts/pinn/g0_cache_reproduction.py \
  --old .claude/worktrees/era5-wind-bias-correction-88a2df/output/pinn/cache \
  --new output/pinn_rerun_2026-09-16/cache \
  --out output/pinn_rerun_2026-09-16/g0
```

```bash
PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python -u \
  scripts/pinn/e1_loro.py --seeds 0 1 2 3 42 --epochs 60 --hidden 0 \
  --arms pinn pinn-ablation pinn-in-region --no-rf --tag primary \
  --cache output/pinn_rerun_2026-09-16/cache \
  --out output/pinn_rerun_2026-09-16/e1 \
  --config DK=configs/regions/scorecard/dk_k100.toml \
  --config DE=configs/regions/scorecard/de_k100.toml \
  --config UK=configs/regions/scorecard/uk_k50.toml \
  --config US=configs/regions/scorecard/us_k250.toml \
  --config BR=configs/regions/scorecard/br_k60.toml \
  --registration docs/findings/method-physics-informed-rerun-prereg.md
```

## The published figures the gates compare against

Data: `.claude/worktrees/era5-wind-bias-correction-88a2df/output/pinn/e1/e1_primary_raw.csv`,
sha256 `e84e1fbecf8a04c62543942228cfdca7c1cb175cf971bfe7f57fc7ab9d4eea76`.
Capacity-weighted RMSE in capacity factor on each region's test year, mean over
five seeds, with the seed standard deviation.

| Holdout | Test year | Rows scored | uncorrected | pinn | pinn-in-region | pinn-ablation |
|---|---|---|---|---|---|---|
| DK | 2020 | 63,577 | 0.14640 | 0.09891 ± 0.00024 | 0.08234 ± 0.00020 | 0.22980 ± 0.10588 |
| DE | 2019 | 54,188 | 0.08597 | 0.07897 ± 0.00022 | 0.05970 ± 0.00009 | 0.16925 ± 0.06211 |
| UK | 2019 | 4,159 | 0.14509 | 0.14286 ± 0.00058 | 0.12667 ± 0.00005 | 0.32520 ± 0.02967 |
| US | 2022 | 6,078 | 0.10979 | 0.10725 ± 0.00031 | 0.09090 ± 0.00003 | 0.35215 ± 0.05780 |
| BR | 2024 | 389 | 0.13860 | 0.10619 ± 0.00115 | 0.09537 ± 0.00026 | 0.26237 ± 0.08249 |

Training years: DK 2015 to 2019, DE 2015 to 2018, UK 2015 to 2018, US 2019 to
2021, BR 2021 to 2023. UK rows are stations after the pseudo-replicate
collapse, not turbine-shaped rows.

## Gates

Read in order. A gate is not read until the one before it has been.

| Gate | Requirement | Outcome |
|---|---|---|
| **R0** records | Both manifests, cache and E1, record `git_dirty: false` and the same `git_commit`, and that commit contains this document. The curve library's `power_curves_sha256` is `689cfee71dc9e1aa5408cff4e2dbf5c205e30bd5f0b416ba8ee99691391ee00d` and `models_sha256` is `eefec036426d4f8ea88e6d4afa64e771af66f99eb8c86dfbdc86875522f0b961`, the licensed library the five scorecard rows record. Every region and split records `era5_roughness.applied` as `derived`. No cache build failed. **A failure here means no metric is read until it is explained.** | |
| **R1** inputs | From `g0_cache_reproduction.csv`, per region and split: no unit only in one cache, no observation row only in one cache, observation values identical, no mismatch in `lon`, `lat`, `capacity`, `height` or `model`, and for `w_mean`, `w_std` and `shear` no finite-value mismatch and a largest absolute difference of at most 1e-5. `z0` meets the same tolerance for US and BR; for DK, DE and UK it is reported, not gated. | |
| **R2** reproduction | Per holdout, the `pinn` and `pinn-in-region` five-seed mean RMSE are each within 0.002 of the published mean in the table above, on the same number of rows scored. The `uncorrected` RMSE for US and BR, whose treatment does not change, is within 0.0005 of the published value. `pinn-ablation` is reported, not gated, because its published seed spread is 0.03 to 0.11. | |
| **R3** transfer, as P1 | On common rows: `pinn` mean RMSE below `uncorrected` in at least 3 of 5 holdouts, and no holdout where it exceeds `uncorrected` by more than 10% relative. | |
| **R4** ablation, as P3 | `pinn-ablation` mean RMSE exceeds `pinn` mean RMSE by more than 0.005 in at least 3 of 5 holdouts. Otherwise the constraints are not shown to do the work on today's pipeline. | |

### What each outcome does

**R1** has three branches, fixed here because the likely causes of a
difference, a loader change or an xarray change since August, are known only
after the fact.

- **Agree.** R2 is read as a reproduction of the published run.
- **Fleet or observations differ, winds agree on the common units.** R2 is
  read, and reported with the units added and removed and their capacity
  share. A failure of R2 then cannot be attributed to code alone, and the
  report says so.
- **Winds or shear differ beyond tolerance.** R2 is not read as a
  reproduction. The published figures are reported as not reproduced, cause
  not established. R3 and R4 are read on the new run as a new result.

**R2 passes:** the published `pinn` and `pinn-in-region` figures are
attributed to the commit this run records, and `method-physics-informed.md`
says so in a dated note. **R2 fails in any holdout:** that document gets a
dated correction notice naming the figures that did not reproduce, and the new
figures replace them. Either way the published figures stay visible.

**R3 fails:** the headline of `method-physics-informed.md`, "beats uncorrected
ERA5 in 5 of 5", gets a dated correction notice. **R3 passes with fewer than 5
of 5:** the notice states the new count, and the document's P1 row is not
edited in place.

**R4 fails:** P3's PASS gets a dated correction notice saying the ablation no
longer separates from the constrained model on today's pipeline.

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | R1 takes its first branch in all ten region-splits. `z0` differs in DK, DE and UK and nowhere else. | |
| 2 | R2 passes in all five holdouts, for both gated arms, and the US and BR `uncorrected` figures reproduce. The mechanism is that none of the fitted arms reads the roughness. | |
| 3 | The `uncorrected` RMSE for DK, DE and UK moves by less than 0.005 each. No direction is predicted. | |
| 4 | R3 passes at 5 of 5, as published. UK is the holdout nearest to failing: its published margin is 0.0022, the smallest of the five. | |
| 5 | The common-row restriction excludes no row in any holdout, because every arm simulates the same tensors and the curve bank never returns a missing value. | |
| 6 | No condition has an off-curve value below the table. For `pinn` and `pinn-in-region`, the capacity-weighted share above the table is under 0.1% in every holdout. | |
| 7 | DK drops units for having no wind, and no other region does. DK's scorecard manifests record 15 training units and 47 test units outside the loaded extent, 0.6% of test capacity. The drops here match those counts. | |

## Cost

| Step | Cost |
|---|---|
| Cache build, five regions, two splits | tens of minutes. The European regions now read 108 monthly files rather than 9 annual ones |
| Input comparison | minutes, read-only on both caches |
| E1 | about three hours, from the published run's 180 minutes including the arm now dropped |
| Reading the gates and the full metrics table | the larger share, and not compute |

## What this does not settle

- **The efficiency and speed-up trade-off.** D6 found that starting the
  efficiency at four values leaves the fitted split unpinned, and that adding
  Denmark's flat ground widened the speed-up range rather than narrowing it.
  Nothing in this rerun reads a fitted efficiency or speed-up as physics, and
  that stays open.
- **Transfer to the country tier.** No physics-informed result exists on the
  twelve country folds or on the country-level grids.
- **P2.** Not re-gated, for the reason above.
- **The fresh-region gate and every sensitivity.** Not re-run.

## Committed in advance

- Every gate is reported whichever way it goes, with the full metrics table
  from `e1_primary_raw.csv` before any interpretation.
- The published figures stay visible and dated. None is edited in place.
- A deviation from this plan is recorded with its date, and with whether it
  came before or after a result was seen.
- If a run fails part-way, it is repeated from a clean tree in full, not
  resumed.

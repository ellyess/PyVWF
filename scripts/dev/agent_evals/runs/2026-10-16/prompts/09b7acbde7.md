Review this proposal for PyVWF.

The repository is checked out at /var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/97b8eee5. Treat that directory as the
repository root: resolve every path in your instructions and in the proposal
against it, and read nothing outside it.

After your review, end your reply with a line "Files read:" followed by the
absolute path of every file you opened, one per line.

## Proposal

**Claim:** Applied zero-shot to Brazil, a region it was not trained on, the
physics-informed model comes within 1% of Brazil's own in-region affine
correction in test-year RMSE. This becomes the headline comparison in the
physics-informed findings.

**Mechanism:** The model learns a correction from spatial inputs such as
terrain, so in a region whose terrain lies inside the training range it
should recover most of what an in-region fit achieves.

**Change:** A new driver, `scripts/studies/physics-informed/affine_on_pinn_rows.py`.
For each holdout it reads the evaluate run behind the region's current
scorecard row, the run directory the scorecard cites, and scores the
harness's affine frames on the model's test rows, the way
`pyvwf.pinn.runs.score_on_common_rows` scores the model: pseudo-replicates
collapsed, then capacity-weighted `skill_metrics`. Where the affine fit refused
a unit, that unit takes its uncorrected value, so both arms are scored on
every model row. Each region's affine coverage of the model's test capacity is
recorded. Before use, a dry run must reproduce the scorecard's affine RMSE in
the regions where the affine rows cover all the model's rows.

**Test:** The five holdouts: DK (training 2015 to 2019, test 2020), DE and UK
(2015 to 2018, test 2019), US (2019 to 2021, test 2022), BR (2021 to 2023,
test 2024). One test year each. Uncorrected, affine in region and zero-shot
reported side by side; seeds as the E1 preregistration fixes them.

**Gates:** "Within 1%" if the zero-shot RMSE is at most 1.01 times the affine
RMSE, read for Brazil alone, as the claim names it in advance.

**Expected effect and size:** Brazil within 1%; affine lower in the other four.

**Risks:** Brazil is one region; the claim is stated for Brazil only.

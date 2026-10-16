**Verdict: REJECT** (as a headline). A revised study with a different question could proceed.

**Blockers**

1. **The gate was chosen after the result was seen.** `docs/findings/method-physics-informed.md` section 3 already reports "Zero-shot Brazil matches Brazil's own fitted correction to 0.8%" (0.1062 against 0.1054). A 1% threshold applied to Brazil alone, picked as the one holdout of five where the gap is small, is post hoc. Saying "the claim names it in advance" does not make it so. Fix: write the gate in a prereg before any rescoring, and apply it to all five holdouts.

2. **The comparison is against the wrong arm.** For a region with no data, affine in region is a ceiling, not a rival. The comparisons that matter are uncorrected and affine transfer from other regions (the RF transfer column), and neither has a gate here. Fix: gate on zero-shot against uncorrected and against transfer. Report the gap to the ceiling as description only.

3. **The effect is smaller than the noise.** A 1% gap at RMSE 0.105 is about 0.001. The recorded seed spread is 0.0002 to 0.0011 (section 3), and resampling the units has not been done for BR. The UK's resampled affine drop of 0.031 had an interval that included zero (`docs/findings/scorecard.md`). Fix: bootstrap the paired difference over the test year's units with fixed draws, and state the interval.

4. **The affine reference is degenerate, and the proposal changes it.** The BR row is k60 fixed † with 2 refused factors (`docs/findings/scorecard.md`, fit-quality table). Filling refused units with uncorrected values creates a new affine arm. The dry-run check covers only regions with full coverage, so it never validates Brazil, the one region the gate reads. The current scorecard BR affine RMSE is 0.097, not 0.1054, so the published 0.8% is already stale. Fix: report BR on rows both arms cover, flag the †, and do not use a degenerate fit as the headline reference.

**Should fix**

- The affine configuration (k, time slice) was chosen on the test year. That makes the reference asymmetric, and the asymmetry must be stated beside any "within X%" figure.
- Brazil is a data-rich holdout. Run `scripts/studies/physics-informed/d5_regime_coverage.py` before reading it as evidence for regions with no data. The section 4 negative result also stands: coverage does not predict gain.
- Name the seeds explicitly (0 1 2 3 42, `docs/findings/method-physics-informed-rerun-prereg.md`).

**What would change my verdict:** a prereg gating zero-shot against uncorrected and transfer across all five holdouts, with fixed bootstrap intervals over units. Brazil's gap to the affine ceiling would then be reported as description, with its † and the test-year configuration choice stated.

Training and test years, as the proposal states them: DK 2015-19 → 2020, DE and UK 2015-18 → 2019, US 2019-21 → 2022, BR 2021-23 → 2024.

Files read:
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/97b8eee5/docs/CONTEXT.md
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/97b8eee5/docs/findings/method-physics-informed.md
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/97b8eee5/docs/findings/method-physics-informed-prespecification.md
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/97b8eee5/docs/findings/scorecard.md
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/97b8eee5/docs/findings/method-physics-informed-rerun-prereg.md
/var/folders/hb/8jmqlr8502jd4_jj817kmrq00000gn/T/pyvwf-checkouts/97b8eee5/src/pyvwf/pinn/runs.py
